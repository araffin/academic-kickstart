"""Plot a selected scalar quantity of grouped SAC TensorBoard runs.

Similar to ``tensorboard_multirun.py``, but restricted to the SAC runs and
with a selectable quantity (``--quantity``, any scalar tag such as
``train/ent_coef`` or ``train/ent_coef_loss``). The runs are grouped by their
experiment folder, interpolated onto a common grid, and plotted with a 95%
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
ENV_NAME = "Isaac-Velocity-Flat-Unitree-A1"
NUM_POINTS = 100
DEFAULT_QUANTITY = "train/ent_coef"

# SAC experiment folder prefix to legend label, in plot order.
SAC_RUN_GROUPS: list[tuple[str, str]] = [
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


def group_run_files(quantity: str) -> list[tuple[str, list[Path]]]:
    """
    Group the run files of one quantity by experiment folder prefix.

    :param quantity: scalar tag to plot, e.g. ``train/ent_coef``
    :return: list of (group label, run file paths) in the order of SAC_RUN_GROUPS
    """
    quantity_file = f"{quantity.replace('/', '__')}.json"
    grouped: list[tuple[str, list[Path]]] = []
    for prefix, label in SAC_RUN_GROUPS:
        run_files = [
            run_dir / quantity_file
            for run_dir in sorted(SCALARS_DIR.iterdir())
            if run_dir.is_dir()
            and run_dir.name.split("__", 1)[0].startswith(prefix)
            and (run_dir / quantity_file).exists()
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
    parser = argparse.ArgumentParser(
        description="Plot a scalar quantity of grouped SAC runs."
    )
    parser.add_argument(
        "--quantity",
        default=DEFAULT_QUANTITY,
        help="scalar tag to plot, e.g. train/ent_coef or train/ent_coef_loss "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--x-axis",
        choices=("time", "steps"),
        default="time",
        help="x axis: wall-clock time in minutes (default) or environment timesteps",
    )
    return parser.parse_args()


def main() -> None:
    """Plot the selected quantity of all SAC run groups."""
    args = parse_args()
    x_column = "Time" if args.x_axis == "time" else "Timesteps"
    x_label = "Time (minutes)" if args.x_axis == "time" else "Timesteps"
    groups = group_run_files(args.quantity)
    if not groups:
        raise SystemExit(f"No SAC runs found for quantity {args.quantity!r}")
    plt.figure(figsize=(12, 8))
    for label, files in groups:
        print(f"{label}: {len(files)} runs")
        plot_group(interpolate_runs(files, x_column), label, x_column)
    plt.xlabel(x_label)
    plt.ylabel(args.quantity)
    plt.title(ENV_NAME)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
