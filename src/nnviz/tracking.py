"""Shared mlflow configuration so training and plotting agree on the store."""

from __future__ import annotations

import mlflow

# mlflow 3.x put the file store into maintenance mode and refuses to track to it, so we
# use a local SQLite backend. Artifacts still land on disk under ./mlruns; plotting
# resolves them through the mlflow client, so the physical path does not matter.
# `make ui` passes the same --backend-store-uri.
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "mnist-hiveplot"


def configure() -> None:
    """Point mlflow at the local file store."""
    mlflow.set_tracking_uri(TRACKING_URI)


def latest_run_id(experiment: str = EXPERIMENT) -> str:
    """Return the most recent run id in the experiment."""
    configure()
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        msg = f"no mlflow experiment named {experiment!r}; run training first"
        raise RuntimeError(msg)
    runs = mlflow.search_runs(
        [exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1
    )
    if len(runs) == 0:
        msg = f"experiment {experiment!r} has no runs; run training first"
        raise RuntimeError(msg)
    return runs.iloc[0]["run_id"]
