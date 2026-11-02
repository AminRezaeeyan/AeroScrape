"""Airflow task callables for the training DAG.

Each DAG run works in its own directory, data/runs/<run_id>/, and tasks pass only file
paths through XCom. Two runs never overwrite each other's files, and a failed run's
inputs stay on disk for debugging until cleanup removes them.
"""

import json
import logging
import os
import re
import shutil
from pathlib import Path

import mlflow
import pandas as pd

from aeroscrape.analysis.association import find_association_rules
from aeroscrape.config import get_app_config, resolve_path
from aeroscrape.data.clean import clean_flights_file, read_flights
from aeroscrape.data.sources import build_dataset
from aeroscrape.data.split import Splits
from aeroscrape.data.validate import validate_raw_data
from aeroscrape.models.baselines import GroupMedianRegressor
from aeroscrape.models.delay_model import log_delay_model
from aeroscrape.models.experiment import (
    TunedParams,
    experiment_lineage,
    make_splits,
    save_outputs,
    train_and_evaluate,
    tune_models,
)
from aeroscrape.models.gate import ModelPredictions, evaluate_gate
from aeroscrape.models.registry import CHALLENGER, CHAMPION, alias_version, client_and_name, promote
from aeroscrape.models.tune import save_best_params

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")


def _raw_data_path(kwargs) -> Path:
    override = kwargs["params"].get("raw_data_path_override")
    return resolve_path(override or get_app_config()["data_paths"]["raw_data"])


def _run_dir(kwargs) -> Path:
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", kwargs["run_id"])
    path = resolve_path(get_app_config()["data_paths"]["runs_dir"]) / safe_run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _git_from_conf(kwargs) -> None:
    """`make train` passes the commit in the run's conf, since containers have no .git directory."""
    conf = kwargs["dag_run"].conf or {}
    if conf.get("git_commit"):
        os.environ["GIT_COMMIT"] = str(conf["git_commit"])
        os.environ["GIT_DIRTY"] = str(conf.get("git_dirty", "unknown"))


def _load_splits(paths: dict[str, str]) -> Splits:
    return Splits(**{name: pd.read_parquet(paths[name]) for name in SPLIT_NAMES})


def _start_mlflow(app_config) -> None:
    mlflow.set_tracking_uri(app_config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(app_config["mlflow"]["experiment_name"])


def _flat_params(params: TunedParams) -> dict[str, object]:
    return {
        f"{variant}.{task}.{key}": value
        for variant, tasks in params.by_variant.items()
        for task, values in tasks.items()
        for key, value in values.items()
    }


def task_validate_raw_data(**kwargs):
    raw_path = _raw_data_path(kwargs)
    logger.info("Validating %s", raw_path)
    validate_raw_data(raw_path)


def task_build_dataset(**kwargs):
    """Clean every flight from the archive and the flight database into this run's directory."""
    cleaned_path = _run_dir(kwargs) / "cleaned.csv"
    if kwargs["params"].get("raw_data_path_override"):
        audit = clean_flights_file(_raw_data_path(kwargs), cleaned_path)
    else:
        audit = build_dataset(get_app_config(), cleaned_path)
    logger.info("Cleaning audit:\n%s", audit.to_markdown())
    kwargs["ti"].xcom_push(key="cleaned_path", value=str(cleaned_path))


def task_split_data(**kwargs):
    """Split cleaned flights by scheduled time and store each split as Parquet."""
    ti = kwargs["ti"]
    app_config = get_app_config()
    flights = read_flights(ti.xcom_pull(task_ids="build_dataset", key="cleaned_path"))
    splits = make_splits(flights, app_config)

    # For testing the gate: train on a random share of train+val; the held-out test window is untouched
    fraction = float(kwargs["params"].get("train_fraction") or 1.0)
    if fraction < 1.0:
        seed = app_config["pipeline"]["random_state"]
        splits = Splits(
            splits.train.sample(frac=fraction, random_state=seed).sort_values("scheduled_datetime"),
            splits.val.sample(frac=fraction, random_state=seed).sort_values("scheduled_datetime"),
            splits.test,
        )
        logger.warning("Training on %.0f%% of train+val (train_fraction=%s)", fraction * 100, fraction)

    out_dir = _run_dir(kwargs) / "splits"
    out_dir.mkdir(exist_ok=True)
    paths = {name: str(out_dir / f"{name}.parquet") for name in SPLIT_NAMES}
    for name in SPLIT_NAMES:
        getattr(splits, name).to_parquet(paths[name], index=False)
    ti.xcom_push(key="split_paths", value=paths)


def task_tune_hyperparameters(**kwargs):
    """Tune with rolling-origin CV on train+val. The test split is never loaded here."""
    ti = kwargs["ti"]
    app_config = get_app_config()
    paths = ti.xcom_pull(task_ids="split_data", key="split_paths")
    train_val = Splits(train=pd.read_parquet(paths["train"]), val=pd.read_parquet(paths["val"]), test=pd.DataFrame())
    params = tune_models(train_val, app_config)

    params_path = str(_run_dir(kwargs) / "best_params.json")
    save_best_params(params.to_dict(), params_path)

    _start_mlflow(app_config)
    with mlflow.start_run(run_name="hyperparameter_tuning"):
        mlflow.set_tag("airflow_run_id", kwargs["run_id"])
        mlflow.log_params(_flat_params(params))
        mlflow.log_metrics(
            {
                f"{variant}.{name}": value
                for variant, scores in params.cv_scores.items()
                for name, value in scores.items()
            }
        )
        mlflow.log_artifact(params_path, artifact_path="best_params")
    ti.xcom_push(key="best_params_path", value=params_path)


def task_train_and_evaluate(**kwargs):
    """Fit, calibrate and evaluate both variants, then log and register the served model."""
    ti = kwargs["ti"]
    _git_from_conf(kwargs)
    app_config = get_app_config()
    pipeline_cfg = app_config["pipeline"]
    splits = _load_splits(ti.xcom_pull(task_ids="split_data", key="split_paths"))
    with open(ti.xcom_pull(task_ids="tune_hyperparameters", key="best_params_path")) as f:
        params = TunedParams.from_dict(json.load(f))

    lineage = experiment_lineage(splits, app_config, _raw_data_path(kwargs))
    result = train_and_evaluate(splits, params, app_config, lineage=lineage)
    report_md = result.report.to_markdown()
    save_outputs(result, _run_dir(kwargs) / "report" / "evaluation.md")
    bundle = result.served

    _start_mlflow(app_config)
    with mlflow.start_run(run_name="train_and_evaluate"):
        # Log the model before the metrics: MLflow 3.1 links the run's existing metrics to a newly
        # logged model and, on a SQLite store, fails with a unique-constraint error when there are many.
        info = log_delay_model(
            bundle,
            example=splits.test.head(20),
            registered_model_name=pipeline_cfg["model_registry_name"],
        )
        mlflow.set_tags(lineage | {"airflow_run_id": kwargs["run_id"], "served_variant": bundle.variant})
        mlflow.log_params(_flat_params(params))
        mlflow.log_params({f"split.{name}_rows": len(getattr(splits, name)) for name in SPLIT_NAMES})
        mlflow.log_param("train_fraction", kwargs["params"].get("train_fraction") or 1.0)
        mlflow.log_params(
            {f"served.{k}": v for k, v in bundle.metadata.items() if k not in ("features", "trained_on", "lineage")}
        )
        mlflow.log_metrics(result.report.to_metrics())
        mlflow.log_text(report_md, "evaluation/report.md")
        mlflow.log_dict(result.report.to_dict(), "evaluation/report.json")
        mlflow.log_dict(bundle.metadata, "evaluation/served_model.json")
        for name, figure in result.figures.items():
            mlflow.log_figure(figure, f"evaluation/{name}.png")
        mlflow.log_artifact(str(resolve_path("config.yaml")), artifact_path="config")

    # The registered version carries the lineage too, so it is visible from the registry
    client, name = client_and_name(app_config)
    version = info.registered_model_version
    for key in ("git_commit", "git_dirty", "training_data_sha256", "training_from", "training_to"):
        client.set_model_version_tag(name, version, key, lineage[key])

    # Registered as the challenger; only the gate can make it the production model
    client.set_registered_model_alias(name, CHALLENGER, version)
    ti.xcom_push(key="challenger_version", value=str(version))
    logger.info("Registered %s version %s as %s", name, version, CHALLENGER)


def _gate_predictions(model_uri: str, frame: pd.DataFrame, name: str) -> ModelPredictions:
    model = mlflow.pyfunc.load_model(model_uri)
    bundle = model.unwrap_python_model().bundle
    out = model.predict(bundle.prepare_input(frame))
    return ModelPredictions(
        name,
        out["p50"].to_numpy(),
        out["p10"].to_numpy(),
        out["p90"].to_numpy(),
        out["p_delay_over_15"].to_numpy(),
    )


def task_evaluate_gate(**kwargs):
    """Score champion and challenger, both loaded from the registry, on the same held-out flights."""
    ti = kwargs["ti"]
    app_config = get_app_config()
    pipeline_cfg, eval_cfg = app_config["pipeline"], app_config["evaluation"]
    splits = _load_splits(ti.xcom_pull(task_ids="split_data", key="split_paths"))
    test = splits.test
    client, name = client_and_name(app_config)
    challenger_version = ti.xcom_pull(task_ids="train_and_evaluate", key="challenger_version")
    champion_version = alias_version(client, name, CHAMPION)

    _start_mlflow(app_config)
    challenger = _gate_predictions(f"models:/{name}/{challenger_version}", test, f"version {challenger_version}")
    champion = None
    notes = []
    if champion_version is not None:
        champion = _gate_predictions(f"models:/{name}/{champion_version}", test, f"version {champion_version}")
        trained_to = client.get_model_version(name, champion_version).tags.get("training_to")
        if trained_to and pd.Timestamp(trained_to) >= test[eval_cfg["time_column"]].min():
            notes.append(
                f"The champion was trained on data up to {trained_to}, which overlaps the gate window, "
                "so the comparison favors the champion."
            )

    baseline = (
        GroupMedianRegressor(["airline", "scheduled_time_of_day"])
        .fit(splits.train_val, splits.train_val[pipeline_cfg["target_regression"]])
        .predict(test)
    )
    result = evaluate_gate(
        test[pipeline_cfg["target_regression"]].to_numpy(),
        test[pipeline_cfg["target_classification"]].to_numpy(),
        challenger,
        champion,
        baseline,
        pipeline_cfg["gate"],
        eval_cfg["bootstrap_samples"],
        eval_cfg["confidence_level"],
        pipeline_cfg["random_state"],
    )
    result.notes += notes
    champion_label = f"version {champion_version}" if champion_version else "no champion"
    table = result.to_markdown(champion_label, f"version {challenger_version}")
    logger.info("\n%s", table)

    client.set_model_version_tag(name, challenger_version, "gate_decision", result.decision)
    with mlflow.start_run(run_name="promotion_gate") as run:
        mlflow.set_tags(
            {
                "airflow_run_id": kwargs["run_id"],
                "challenger_version": challenger_version,
                "champion_version": champion_version or "none",
                "decision": result.decision,
            }
        )
        mlflow.log_metrics(result.metrics)
        mlflow.log_text(table, "gate/comparison.md")
    ti.xcom_push(key="decision", value=result.decision)
    ti.xcom_push(key="gate_run_id", value=run.info.run_id)


def choose_branch(**kwargs) -> str:
    decision = kwargs["ti"].xcom_pull(task_ids="evaluate_gate", key="decision")
    return "promote" if decision == "promote" else "reject"


def _record_action(kwargs, action: str) -> None:
    run_id = kwargs["ti"].xcom_pull(task_ids="evaluate_gate", key="gate_run_id")
    _start_mlflow(get_app_config())
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("action", action)


def task_promote(**kwargs):
    client, name = client_and_name(get_app_config())
    version = kwargs["ti"].xcom_pull(task_ids="train_and_evaluate", key="challenger_version")
    promote(client, name, version, reason=f"passed gate in {kwargs['run_id']}")
    _record_action(kwargs, f"promoted version {version} to production")


def task_reject(**kwargs):
    version = kwargs["ti"].xcom_pull(task_ids="train_and_evaluate", key="challenger_version")
    _record_action(kwargs, f"kept the champion; version {version} stays challenger")
    logger.info("Challenger version %s rejected by the gate; production unchanged", version)


def task_run_association_mining(**kwargs):
    """FP-Growth association rules over the cleaned departures (exploratory, not used by the models)."""
    config = get_app_config()
    cleaned_path = kwargs["ti"].xcom_pull(task_ids="build_dataset", key="cleaned_path")
    output_path = _run_dir(kwargs) / config["association_mining"]["results"]["association_rules_filename"]
    find_association_rules(
        data_path=cleaned_path,
        output_path=output_path,
        min_support=config["association_mining"]["min_support"],
        min_threshold=config["association_mining"]["min_threshold"],
    )
    results_dir = resolve_path(config["association_mining"]["results"]["path"])
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(output_path, results_dir / output_path.name)


def task_cleanup_processed_data(**kwargs):
    """Remove this run's intermediate files; models and reports live in MLflow."""
    run_dir = _run_dir(kwargs)
    shutil.rmtree(run_dir, ignore_errors=True)
    logger.info("Removed %s", run_dir)
