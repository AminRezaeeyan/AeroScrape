import json
import logging
import os

import joblib
import mlflow
import pandas as pd
from mlflow import MlflowClient

from aeroscrape.analysis.association import find_association_rules
from aeroscrape.config import get_app_config, resolve_path
from aeroscrape.config import project_root as _project_root
from aeroscrape.data.clean import clean_flights_file, read_flights
from aeroscrape.data.split import Splits
from aeroscrape.data.validate import validate_raw_data
from aeroscrape.models.experiment import TunedParams, make_splits, train_and_evaluate, tune_models
from aeroscrape.models.train import save_metrics, save_model
from aeroscrape.models.tune import save_best_params

logger = logging.getLogger(__name__)


def get_project_root():
    return str(_project_root())


def task_validate_raw_data(**kwargs):
    logger.info("Starting task: Validate Raw Data")
    project_root = get_project_root()
    app_config = get_app_config()

    raw_data_path_override = kwargs["params"].get("raw_data_path_override")
    if raw_data_path_override:
        raw_data_path = os.path.join(project_root, raw_data_path_override)
        logger.info(f"Using overridden raw_data_path from DAG params: {raw_data_path}")
    else:
        raw_data_path = os.path.join(project_root, app_config["data_paths"]["raw_data"])
        logger.info(f"Using raw_data_path from config: {raw_data_path}")

    validate_raw_data(raw_data_path)
    logger.info("Raw data validation task completed successfully.")


def task_clean_and_engineer_data(**kwargs):
    logger.info("Starting task: Clean and Engineer Data")
    project_root = get_project_root()
    app_config = get_app_config()

    raw_data_path_override = kwargs["params"].get("raw_data_path_override")
    if raw_data_path_override:
        raw_data_path = os.path.join(project_root, raw_data_path_override)
        logger.info(f"Using overridden raw_data_path from DAG params for cleaning: {raw_data_path}")
    else:
        raw_data_path = os.path.join(project_root, app_config["data_paths"]["raw_data"])
        logger.info(f"Using raw_data_path from config for cleaning: {raw_data_path}")

    cleaned_data_path = os.path.join(project_root, app_config["data_paths"]["cleaned_data"])
    os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)

    audit = clean_flights_file(raw_data_path, cleaned_data_path)
    logger.info("Cleaning audit:\n%s", audit.to_markdown())
    logger.info("Data cleaning and feature engineering complete.")


SPLIT_NAMES = ("train", "val", "test")


def _splits_dir(app_config) -> str:
    return os.path.join(resolve_path(app_config["data_paths"]["processed_data_dir"]), "splits")


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


def task_split_data(**kwargs):
    """Split cleaned flights by scheduled time and store each split as Parquet."""
    app_config = get_app_config()
    flights = read_flights(resolve_path(app_config["data_paths"]["cleaned_data"]))
    splits = make_splits(flights, app_config)

    out_dir = _splits_dir(app_config)
    os.makedirs(out_dir, exist_ok=True)
    paths = {name: os.path.join(out_dir, f"{name}.parquet") for name in SPLIT_NAMES}
    for name in SPLIT_NAMES:
        getattr(splits, name).to_parquet(paths[name], index=False)
    kwargs["ti"].xcom_push(key="split_paths", value=paths)


def task_tune_hyperparameters(**kwargs):
    """Tune with rolling-origin CV on train+val. The test split is never loaded here."""
    ti = kwargs["ti"]
    app_config = get_app_config()
    paths = ti.xcom_pull(task_ids="split_data_task", key="split_paths")
    train_val = Splits(train=pd.read_parquet(paths["train"]), val=pd.read_parquet(paths["val"]), test=pd.DataFrame())
    params = tune_models(train_val, app_config)

    params_path = str(resolve_path(app_config["data_paths"]["best_params"]))
    save_best_params(params.to_dict(), params_path)

    _start_mlflow(app_config)
    with mlflow.start_run(run_name="hyperparameter_tuning"):
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
    """Fit both models on train+val, compare them with baselines on test, and register them."""
    ti = kwargs["ti"]
    app_config = get_app_config()
    pipeline_cfg = app_config["pipeline"]
    splits = _load_splits(ti.xcom_pull(task_ids="split_data_task", key="split_paths"))
    with open(ti.xcom_pull(task_ids="tune_hyperparameters_task", key="best_params_path")) as f:
        params = TunedParams.from_dict(json.load(f))

    result = train_and_evaluate(splits, params, app_config)
    report_md = result.report.to_markdown()
    report_path = resolve_path(app_config["evaluation"]["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")

    # The API loads the bare estimator from the registry and this preprocessor from disk
    preprocessor_path = os.path.join(
        resolve_path(pipeline_cfg["model_output_dir"]), pipeline_cfg["preprocessor_filename"]
    )
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
    joblib.dump(result.regressor.named_steps["preprocess"], preprocessor_path)
    regressor = result.regressor.named_steps["model"]
    classifier = result.classifier.named_steps["model"]

    _start_mlflow(app_config)
    with mlflow.start_run(run_name="train_and_evaluate"):
        mlflow.log_params(_flat_params(params))
        mlflow.log_param("served_variant", result.served_variant)
        mlflow.log_params({f"split.{k}": app_config["evaluation"][k] for k in ("train_end", "val_end")})
        mlflow.log_metrics(result.report.to_metrics())
        mlflow.log_text(report_md, "evaluation/report.md")
        mlflow.log_dict(result.report.to_dict(), "evaluation/report.json")
        mlflow.log_artifact(str(resolve_path("config.yaml")), artifact_path="config")
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

        reg_info = mlflow.sklearn.log_model(
            sk_model=regressor,
            name="lgbm-regressor",
            registered_model_name=pipeline_cfg["model_registry_name_regression"],
        )
        mlflow.sklearn.log_model(
            sk_model=classifier,
            name="logistic-classifier",
            registered_model_name=pipeline_cfg["model_registry_name_classification"],
        )

    # Every run still becomes the served model; a champion/challenger gate replaces this in Phase 5
    alias = pipeline_cfg.get("model_alias_for_api", "production")
    MlflowClient().set_registered_model_alias(
        pipeline_cfg["model_registry_name_regression"], alias, reg_info.registered_model_version
    )
    logger.info("Set alias '%s' to regressor version %s", alias, reg_info.registered_model_version)

    save_model(regressor, "lgbm_regressor", subfolder="regression")
    save_model(classifier, "logistic_classifier", subfolder="classification")
    save_metrics(result.report.to_metrics(), "evaluation", subfolder="")


def task_run_association_mining():
    """
    Airflow task to run the FP-Growth association Rule mining Task
    """
    logging.info("--- Starting Association Rule Mining Task ---")
    config = get_app_config()
    project_root = get_project_root()  # <-- Added this

    # <-- Added project_root to these paths to make them absolute
    cleaned_data_path = os.path.join(project_root, config["data_paths"]["cleaned_data"])
    results_dir = os.path.join(project_root, config["association_mining"]["results"]["path"])
    output_path = os.path.join(results_dir, config["association_mining"]["results"]["association_rules_filename"])

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    find_association_rules(
        data_path=cleaned_data_path,
        output_path=output_path,
        min_support=config["association_mining"]["min_support"],
        min_threshold=config["association_mining"]["min_threshold"],
    )
    logging.info("--- Association Rule Mining Task Completed ---")


def task_cleanup_processed_data(**kwargs):
    ti = kwargs["ti"]
    split_paths = ti.xcom_pull(task_ids="split_data_task", key="split_paths") or {}
    params_path = ti.xcom_pull(task_ids="tune_hyperparameters_task", key="best_params_path")
    for path in [*split_paths.values(), params_path]:
        if path and os.path.exists(path):
            os.remove(path)
            logger.info("Removed intermediate file: %s", path)
