"""The DAGs import cleanly and have the structure the pipeline relies on."""

import pytest

airflow = pytest.importorskip("airflow")


@pytest.fixture(scope="module")
def dagbag(tmp_path_factory):
    import os

    os.environ.setdefault("AIRFLOW_HOME", str(tmp_path_factory.mktemp("airflow")))
    os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
    from airflow.models import DagBag

    return DagBag("dags", include_examples=False)


def test_no_import_errors(dagbag) -> None:
    assert dagbag.import_errors == {}
    assert set(dagbag.dag_ids) == {"ingest_flights", "train_delay_model", "monitor_model"}


def test_training_ends_in_a_gate_branch(dagbag) -> None:
    dag = dagbag.dags["train_delay_model"]
    assert "choose_branch" in dag.get_task("evaluate_gate").downstream_task_ids
    assert dag.get_task("choose_branch").downstream_task_ids >= {"promote", "reject"}


@pytest.mark.parametrize("dag_id", ["ingest_flights", "train_delay_model", "monitor_model"])
def test_a_failed_task_fails_the_run(dagbag, dag_id: str) -> None:
    dag = dagbag.dags[dag_id]
    watcher = dag.get_task("watcher")
    assert watcher.trigger_rule == "one_failed"
    main_tasks = {t.task_id for t in dag.tasks} - {"watcher", "cleanup"}
    assert watcher.upstream_task_ids == main_tasks


def test_training_can_run_on_either_data_source(dagbag) -> None:
    source = dagbag.dags["train_delay_model"].params.get_param("source")
    assert source.value == "iran_fids"
    assert set(source.schema["enum"]) == {"iran_fids", "bts"}


def test_monitoring_can_only_trigger_training_through_its_decision(dagbag) -> None:
    dag = dagbag.dags["monitor_model"]
    assert dag.get_task("trigger_retraining").upstream_task_ids == {"decide_retrain"}
    assert dag.get_task("decide_retrain").upstream_task_ids == {"live_performance", "detect_drift"}
    assert dag.get_task("trigger_retraining").trigger_dag_id == "train_delay_model"
