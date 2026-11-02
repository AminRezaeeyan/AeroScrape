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
    assert set(dagbag.dag_ids) == {"ingest_flights", "train_delay_model"}


def test_training_ends_in_a_gate_branch(dagbag) -> None:
    dag = dagbag.get_dag("train_delay_model")
    assert "choose_branch" in dag.get_task("evaluate_gate").downstream_task_ids
    assert dag.get_task("choose_branch").downstream_task_ids >= {"promote", "reject"}


@pytest.mark.parametrize("dag_id", ["ingest_flights", "train_delay_model"])
def test_a_failed_task_fails_the_run(dagbag, dag_id: str) -> None:
    dag = dagbag.get_dag(dag_id)
    watcher = dag.get_task("watcher")
    assert watcher.trigger_rule == "one_failed"
    main_tasks = {t.task_id for t in dag.tasks} - {"watcher", "cleanup"}
    assert watcher.upstream_task_ids == main_tasks
