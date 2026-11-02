from airflow.exceptions import AirflowFailException


def fail_run():
    """Watcher task: runs only if an upstream task failed, so the DAG run is marked failed.

    Cleanup tasks run with trigger_rule="all_done" and are leaves, so without this a run
    whose main task failed would still be marked successful.
    """
    raise AirflowFailException("An upstream task failed; see its logs.")
