"""Stimulus sequence parsing and FBCCA score organisation."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def parse_sequence(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse the MATLAB-format sequence file.

    Returns
    -------
    sequence : np.ndarray, shape (100, 2), dtype int
        Each row is (red_index, green_index), 1-indexed.
    red_array : np.ndarray, shape (10,)
        Red channel PWM intensity values (D/A units).
    green_array : np.ndarray, shape (10,)
        Green channel PWM intensity values (D/A units).
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        m = re.match(r"^([A-Za-z]\w*)\s*=\s*$", stripped)
        if m:
            current = m.group(1)
            sections[current] = []
        elif current is not None and stripped:
            sections[current].append(stripped)

    sequence = np.array(
        [list(map(int, row.split())) for row in sections["sequence1"]],
        dtype=int,
    )

    def _parse_array(lines: list[str]) -> np.ndarray:
        multiplier = 1.0
        values: list[float] = []
        for line in lines:
            if line.endswith("*"):
                multiplier = float(line[:-1].strip())
            else:
                values.extend(map(float, line.split()))
        return np.array(values) * multiplier

    red_array = _parse_array(sections["redArray"])
    green_array = _parse_array(sections["greenArray"])

    return sequence, red_array, green_array


def split_baseline_task(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split FBCCA scores into baseline and task epochs.

    Epoch numbering is 1-indexed per the experimental protocol:
    - Baselines : epochs 1, 2, 103, 104
    - Task      : epochs 3–102  (100 stimuli)

    Parameters
    ----------
    scores : np.ndarray, shape (104,)

    Returns
    -------
    baseline : np.ndarray, shape (4,)
    task     : np.ndarray, shape (100,)
    """
    baseline = scores[[0, 1, 102, 103]]
    task = scores[2:102]
    return baseline, task


def build_stimulus_matrix(
    task_scores: np.ndarray,
    sequence: np.ndarray,
) -> np.ndarray:
    """Map 100 task FBCCA scores onto a 10×10 (green × red) grid.

    Parameters
    ----------
    task_scores : np.ndarray, shape (100,)
        FBCCA scores for task epochs 3–102, in presentation order.
    sequence : np.ndarray, shape (100, 2), dtype int
        Columns are (red_index, green_index), 1-indexed.

    Returns
    -------
    matrix : np.ndarray, shape (10, 10)
        matrix[green_idx-1, red_idx-1] = FBCCA score.
        Row 0 = lowest green intensity; column 0 = lowest red intensity.
    """
    matrix = np.zeros((10, 10))
    for score, (red_idx, green_idx) in zip(task_scores, sequence):
        matrix[green_idx - 1, red_idx - 1] = score
    return matrix
