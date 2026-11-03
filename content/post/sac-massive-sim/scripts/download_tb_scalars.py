"""Download selected scalar tags from the TensorBoard data server.

Fetches the list of runs from the TensorBoard data server, downloads the
scalar series for the tags below, and stores one JSON file per (run, tag) pair
under ``data/scalars/<run>/<tag>.json``.

- ``rollout/ep_rew_mean`` is downloaded for every run.
- ``train/ent_coef`` and ``train/ent_coef_loss`` are downloaded for the SAC
  runs only.
"""

import json
from pathlib import Path

import requests  # type: ignore[import-untyped]

TB_BASE_URL = "http://rmc-lx0695:6006"
DATA_DIR = Path(__file__).resolve().parent / "data"
SCALARS_OUT_DIR = DATA_DIR / "scalars"

TAGS_FOR_ALL_RUNS = ("rollout/ep_rew_mean",)
TAGS_FOR_SAC_RUNS = ("train/ent_coef", "train/ent_coef_loss")

SeriesByRun = dict[str, list[dict[str, float]]]


def is_sac_run(run_name: str) -> bool:
    """
    Check whether a run is a SAC run (its last path element starts with SAC).

    :param run_name: full relative run path, e.g. ``exp/env/date/SAC_1``
    :return: True if the run is a SAC run
    """
    return run_name.rsplit("/", 1)[-1].startswith("SAC")


def load_run_names() -> list[str]:
    """
    Fetch the list of run names from the TensorBoard data server.

    :return: the run names in the order they were logged
    """
    response = requests.get(f"{TB_BASE_URL}/data/runs", timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_tag_series(tag: str) -> SeriesByRun:
    """
    Fetch one scalar tag for every run from the TensorBoard data server.

    :param tag: scalar tag name, e.g. ``rollout/ep_rew_mean``
    :return: mapping from run name to its list of ``{wallTime, step, value}`` records
    """
    request_body = {
        "requests": json.dumps([{"plugin": "scalars", "tag": tag, "runId": None}])
    }
    response = requests.post(
        f"{TB_BASE_URL}/data/plugin/timeseries/timeSeries",
        data=request_body,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()[0]["runToSeries"]


def sanitize_for_path(name: str) -> str:
    """
    Make a run or tag name usable as a single path element.

    :param name: run or tag name containing ``/`` separators
    :return: the name with ``/`` replaced by ``__``
    """
    return name.replace("/", "__")


def download_tag(tag: str, run_names: list[str]) -> None:
    """
    Download one scalar tag and store it per run under ``data/scalars/``.

    :param tag: scalar tag name
    :param run_names: run names to store this tag for
    """
    series_by_run = fetch_tag_series(tag)
    for run_name in run_names:
        if run_name not in series_by_run:
            raise RuntimeError(f"run {run_name!r} has no data for tag {tag!r}")
        series = series_by_run[run_name]
        out_file = (
            SCALARS_OUT_DIR
            / sanitize_for_path(run_name)
            / f"{sanitize_for_path(tag)}.json"
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(series, indent=1) + "\n")
        print(f"{out_file.relative_to(DATA_DIR)}: {len(series)} points")


def main() -> None:
    """Download all requested scalar tags for all runs."""
    all_run_names = load_run_names()
    sac_run_names = [run_name for run_name in all_run_names if is_sac_run(run_name)]
    print(f"{len(all_run_names)} runs, {len(sac_run_names)} of them SAC")
    for tag in TAGS_FOR_ALL_RUNS:
        download_tag(tag, all_run_names)
    for tag in TAGS_FOR_SAC_RUNS:
        download_tag(tag, sac_run_names)


if __name__ == "__main__":
    main()
