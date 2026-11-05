"""Utilities for the hierarchical 8-command MI formulation.

The proposed Joint8 decoder combines two predictions made from the same
physical trial:

    early 0--2.5 s -> four-class MI identity (task4)
    late  2.5--5 s -> binary temporal state (state2)

The final zero-based command is ``task4 + 4 * state2``.

This module intentionally depends only on NumPy/Pandas/scikit-learn so the
label and cross-validation logic can be tested independently of MOABB and the
classifier implementations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


PAIR_OFFSETS = 4


def validate_joint_labels(y_task, y_state, y_joint) -> None:
    """Validate the three zero-based label arrays used by Joint8."""
    y_task = np.asarray(y_task, dtype=int).reshape(-1)
    y_state = np.asarray(y_state, dtype=int).reshape(-1)
    y_joint = np.asarray(y_joint, dtype=int).reshape(-1)

    n = len(y_joint)
    if len(y_task) != n or len(y_state) != n:
        raise ValueError(
            "Joint8 label arrays have different lengths: "
            f"task={len(y_task)}, state={len(y_state)}, joint={len(y_joint)}."
        )

    if not np.all(np.isin(y_task, [0, 1, 2, 3])):
        raise ValueError("Task4 labels must be zero-based values 0..3.")
    if not np.all(np.isin(y_state, [0, 1])):
        raise ValueError("State2 labels must be zero-based values 0..1.")
    if not np.all(np.isin(y_joint, np.arange(8))):
        raise ValueError("Joint8 labels must be zero-based values 0..7.")

    expected = y_task + 4 * y_state
    if not np.array_equal(y_joint, expected):
        bad = np.flatnonzero(y_joint != expected)[:10]
        raise ValueError(
            "Joint8 mapping is inconsistent with task + 4*state. "
            f"First mismatching indices: {bad.tolist()}"
        )


def _normalise_meta(meta, n_trials: int) -> pd.DataFrame:
    """Return a reset-index metadata frame with stable session/run columns."""
    if meta is None:
        df = pd.DataFrame(index=np.arange(n_trials))
    elif isinstance(meta, pd.DataFrame):
        df = meta.copy().reset_index(drop=True)
    else:
        df = pd.DataFrame(meta).reset_index(drop=True)

    if len(df) != n_trials:
        raise ValueError(
            f"Metadata has {len(df)} rows but there are {n_trials} trials."
        )

    # MOABB normally supplies these columns.  For utilities/tests that do not,
    # use explicit single-session/single-run placeholders rather than silently
    # deriving groups from row number alone.
    if "session" not in df.columns:
        df["session"] = "1"
    if "run" not in df.columns:
        df["run"] = "1"

    df["session"] = df["session"].astype(str)
    df["run"] = df["run"].astype(str)
    return df


def add_joint_metadata(meta, y_task, y_state, y_joint) -> pd.DataFrame:
    """Attach stable trial/label columns to MOABB metadata."""
    y_task = np.asarray(y_task, dtype=int).reshape(-1)
    y_state = np.asarray(y_state, dtype=int).reshape(-1)
    y_joint = np.asarray(y_joint, dtype=int).reshape(-1)
    validate_joint_labels(y_task, y_state, y_joint)

    df = _normalise_meta(meta, len(y_joint))
    df["trial_index"] = np.arange(len(df), dtype=int)
    df["trial_in_run"] = df.groupby(["session", "run"]).cumcount().astype(int)
    df["raw_label"] = y_joint + 1
    df["task_label"] = y_task
    df["state_label"] = y_state
    df["joint_label"] = y_joint
    return df


def build_pair_groups(meta, y_joint, strict: bool = True):
    """Build one group ID for each adjacent sustained/MI-to-Rest pair.

    Expected zero-based Joint8 pairs are ``0->4, 1->5, 2->6, 3->7`` within
    each run.  In strict mode, any odd number of trials or mismatching adjacent
    pair raises an error.  This prevents an apparently "pair-aware" CV from
    silently grouping unrelated trials.
    """
    y_joint = np.asarray(y_joint, dtype=int).reshape(-1)
    df = _normalise_meta(meta, len(y_joint))
    groups = np.empty(len(df), dtype=object)

    for (session, run), idx_values in df.groupby(["session", "run"], sort=False).groups.items():
        idx = np.asarray(list(idx_values), dtype=int)
        labels = y_joint[idx]

        if len(idx) % 2 != 0:
            msg = (
                f"session={session}, run={run} has {len(idx)} trials; "
                "cannot form complete adjacent pairs."
            )
            if strict:
                raise ValueError(msg)

        pair_no = 0
        pos = 0
        while pos < len(idx):
            if pos + 1 >= len(idx):
                groups[idx[pos]] = f"s{session}|r{run}|singleton{pair_no:04d}"
                break

            first = int(labels[pos])
            second = int(labels[pos + 1])
            is_expected = first in (0, 1, 2, 3) and second == first + PAIR_OFFSETS

            if not is_expected and strict:
                raise ValueError(
                    "Unexpected adjacent Joint8 pair in "
                    f"session={session}, run={run}, trials={pos}/{pos+1}: "
                    f"labels={first}/{second} (raw={first+1}/{second+1})."
                )

            # Even in non-strict mode the two observed neighbours are kept
            # together; callers can separately inspect pair_valid.
            gid = f"s{session}|r{run}|pair{pair_no:04d}"
            groups[idx[pos]] = gid
            groups[idx[pos + 1]] = gid
            pair_no += 1
            pos += 2

    return groups


def make_cv_splits(
    y_joint,
    meta=None,
    cv_mode: str = "trial",
    n_splits: int = 5,
    random_state: int = 42,
):
    """Create reproducible CV splits for the hierarchical Joint8 runner.

    Parameters
    ----------
    cv_mode : {"trial", "pair"}
        ``trial`` reproduces conventional shuffled StratifiedKFold.
        ``pair`` keeps each adjacent sustained/MI-to-Rest pair in one fold via
        StratifiedGroupKFold.
    """
    y_joint = np.asarray(y_joint, dtype=int).reshape(-1)
    dummy_x = np.zeros((len(y_joint), 1), dtype=float)

    if cv_mode == "trial":
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        return list(splitter.split(dummy_x, y_joint)), None

    if cv_mode == "pair":
        groups = build_pair_groups(meta, y_joint, strict=True)
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = list(splitter.split(dummy_x, y_joint, groups=groups))

        # Defensive check: no pair may occur in both train and test.
        for fold, (train_idx, test_idx) in enumerate(splits):
            overlap = set(groups[train_idx]).intersection(groups[test_idx])
            if overlap:
                raise RuntimeError(
                    f"Pair leakage detected in fold {fold}: {sorted(overlap)[:5]}"
                )
        return splits, groups

    raise ValueError("cv_mode must be 'trial' or 'pair'.")
