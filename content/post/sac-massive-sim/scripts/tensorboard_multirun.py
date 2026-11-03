"""Plot the mean episode return of grouped TensorBoard runs.

Loads the per-run scalar files downloaded by ``download_tb_scalars.py`` from
``data/scalars/``, groups the runs by their experiment folder, interpolates
the runs of each group onto a common grid, and plots them with a 95%
confidence interval. The x axis can be wall-clock time (default) or
environment timesteps (``--x-axis steps``).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns
from scipy import interpolate  # type: ignore[import-untyped]

sns.set_theme(font_scale=1.5)
sns.set_style("whitegrid")

SCALARS_DIR = Path(__file__).resolve().parent / "data" / "scalars"
EP_REW_MEAN_TAG = "rollout__ep_rew_mean.json"
ENV_NAME = "Isaac-Velocity-Flat-Unitree-A1"
NUM_POINTS = 100

# Experiment folder prefix to legend label, in plot order.
RUN_GROUPS: list[tuple[str, str]] = [
    ("ppo_baseline_4096", "PPO"),
    ("sac_limits_2048", "SAC (w/ limits)"),
    ("sac_big_net_limits_2048", "SAC (w/ limits + bigger net)"),
    ("sac_ent_coef_2048", "SAC (w/ limits + ent coef)"),
    ("sac_big_net_ent_coef_2048", "SAC (w/ limits + ent coef + bigger net)"),
    ("sac_baseline_2048", "SAC (baseline, original action space)"),
]


def load_data(filename: Path) -> pd.DataFrame:
    """
    Load one run's scalar series from a JSON file.

    :param filename: path to a JSON file with a list of
        ``{"wallTime": ..., "step": ..., "value": ...}`` records
    :return: DataFrame with ``Time`` (minutes since the first record),
        ``Timesteps`` and ``Mean Episode Return`` columns
    """
    records = json.loads(filename.read_text())
    df = pd.DataFrame.from_records(records, columns=["wallTime", "step", "value"])
    df = df.rename(
        columns={
            "wallTime": "Timestamp",
            "step": "Timesteps",
            "value": "Mean Episode Return",
        }
    )
    df["Time"] = (df["Timestamp"] - df["Timestamp"].iloc[0]) / 60
    return df


def group_run_files() -> list[tuple[str, list[Path]]]:
    """
    Group the run files by experiment folder prefix.

    :return: list of (group label, run file paths) in the order of RUN_GROUPS
    """
    grouped: list[tuple[str, list[Path]]] = []
    for prefix, label in RUN_GROUPS:
        run_files = [
            run_dir / EP_REW_MEAN_TAG
            for run_dir in sorted(SCALARS_DIR.iterdir())
            if run_dir.is_dir() and run_dir.name.split("__", 1)[0].startswith(prefix)
        ]
        if run_files:
            grouped.append((label, run_files))
    return grouped


def interpolate_runs(
    files: list[Path], x_column: str, num_points: int = NUM_POINTS
) -> pd.DataFrame:
    """
    Load multiple runs and interpolate them onto a common grid.

    :param files: paths to the run JSON files
    :param x_column: column used as x axis (``Time`` or ``Timesteps``)
    :param num_points: number of points of the common grid
    :return: DataFrame with ``x_column``, ``Mean Episode Return`` and ``Run`` columns
    """
    runs_data = [load_data(filename) for filename in files]
    min_x = max(df[x_column].min() for df in runs_data)
    max_x = min(df[x_column].max() for df in runs_data)
    x_grid = np.linspace(min_x, max_x, num_points)

    interpolated_data = []
    for run_index, df in enumerate(runs_data):
        interpolate_run = interpolate.interp1d(
            df[x_column],
            df["Mean Episode Return"],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        interpolated_data.append(
            pd.DataFrame(
                {
                    x_column: x_grid,
                    "Mean Episode Return": interpolate_run(x_grid),
                    "Run": run_index + 1,
                }
            )
        )
    return pd.concat(interpolated_data, ignore_index=True)


def plot_group(data: pd.DataFrame, label: str, x_column: str) -> None:
    """
    Plot one group of runs with a 95% confidence interval.

    :param data: interpolated data of the group
    :param label: legend label of the group
    :param x_column: column used as x axis
    """
    sns.lineplot(
        data=data,
        x=x_column,
        y="Mean Episode Return",
        errorbar=("ci", 95),
        linewidth=2,
        label=label,
    )


def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments.

    :return: the parsed arguments
    """
    parser = argparse.ArgumentParser(description="Plot grouped TensorBoard runs.")
    parser.add_argument(
        "--x-axis",
        choices=("time", "steps"),
        default="time",
        help="x axis: wall-clock time in minutes (default) or environment timesteps",
    )
    parser.add_argument(
        "--y-min", type=float, default=None, help="lower y axis limit (default: auto)"
    )
    parser.add_argument(
        "--y-max", type=float, default=None, help="upper y axis limit (default: auto)"
    )
    return parser.parse_args()


def main() -> None:
    """Plot the mean episode return of all run groups."""
    args = parse_args()
    x_column = "Time" if args.x_axis == "time" else "Timesteps"
    x_label = "Time (minutes)" if args.x_axis == "time" else "Timesteps"
    plt.figure(figsize=(12, 8))
    for label, files in group_run_files():
        print(f"{label}: {len(files)} runs")
        plot_group(interpolate_runs(files, x_column), label, x_column)
    if args.y_min is not None or args.y_max is not None:
        plt.ylim(bottom=args.y_min, top=args.y_max)
    plt.xlabel(x_label)
    plt.ylabel("Mean Episode Return")
    plt.title(ENV_NAME)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
