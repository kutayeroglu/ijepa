#!/usr/bin/env python3
"""
Visualize pre-training iteration speed against color mask ratio.

Input CSV format:
  - Column 1: Color Mask Ratio (%), used as x-axis
  - Column 2: Iteration speed, used as y-axis

Example:
  python visualization/visualize_iter_speed.py \
      --input data/iter_speed.csv \
      --output visualization/iter_speed.png
"""

from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Color Mask Ratio (%) vs Pre-training Iteration Speed."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input CSV file. Column 1 is x and column 2 is y.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. If omitted, the figure is shown interactively.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_false",
        dest="annotate",
        help="Disable point annotations (enabled by default).",
    )
    parser.set_defaults(annotate=True)
    return parser.parse_args()


def _looks_like_header(row: list[str]) -> bool:
    text = " ".join(cell.strip() for cell in row).lower()
    header_tokens = ("ratio", "mask", "speed", "iter", "x", "y", "value")
    return any(token in text for token in header_tokens)


def load_xy_from_csv(path: Path) -> tuple[list[float], list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    x_values: list[float] = []
    y_values: list[float] = []

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        for line_no, row in enumerate(reader, start=1):
            if not row:
                continue

            row = [cell.strip() for cell in row]
            if not any(row):
                continue
            if row[0].startswith("#"):
                continue
            if len(row) < 2:
                raise ValueError(f"Row {line_no} has fewer than two columns: {row}")

            try:
                x_val = float(row[0])
                y_val = float(row[1])
            except ValueError:
                if not x_values and _looks_like_header(row):
                    continue
                raise ValueError(
                    f"Row {line_no} is not numeric in first two columns: {row}"
                ) from None

            y_values.append(y_val)
            x_values.append(x_val)

    if not x_values:
        raise ValueError(f"No valid numeric data rows found in file: {path}")

    return x_values, y_values


def plot_iteration_speed(
    x_values: list[float],
    y_values: list[float],
    output_path: Path | None = None,
    title: str | None = None,
    annotate: bool = False,
) -> None:
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to plot figures. Install it with: "
            "pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        x_values,
        y_values,
        marker="x",
        linewidth=1.2,
        label="Pre-training Iteration Speed",
    )

    if annotate:
        for x_val, y_val in zip(x_values, y_values):
            ax.annotate(
                f"{y_val}",
                (x_val, y_val),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
            )

    ax.set_xlabel("Color Mask Ratio (%)")
    ax.set_ylabel("Pre-training Iteration Speed")
    if title:
        ax.set_title(title)

    ax.set_ylim(top=1.35)
    # Show only provided color mask ratio values on the x-axis.
    ax.set_xticks(sorted(set(x_values)))
    ax.legend(loc="upper left")
    ax.grid(False)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    plt.show()


def main() -> None:
    args = parse_args()
    x_values, y_values = load_xy_from_csv(args.input)
    plot_iteration_speed(
        x_values=x_values,
        y_values=y_values,
        output_path=args.output,
        title=args.title,
        annotate=args.annotate,
    )


if __name__ == "__main__":
    main()
