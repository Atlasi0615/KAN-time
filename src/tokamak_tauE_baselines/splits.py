from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass
class SplitFrames:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def _split_by_top_target(
    df: pd.DataFrame,
    target_col: str,
    holdout_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 2:
        raise ValueError("Need at least 2 rows to build a train/validation split.")

    holdout_n = max(1, min(int(holdout_size), len(df) - 1))
    ranked = df.sort_values(target_col, ascending=False, kind="mergesort").reset_index(drop=True)
    holdout_df = ranked.iloc[:holdout_n].copy()
    remain_df = ranked.iloc[holdout_n:].copy()
    return remain_df.reset_index(drop=True), holdout_df.reset_index(drop=True)


def random_split(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> SplitFrames:
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    rel_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=rel_val_size,
        random_state=seed,
        shuffle=True,
    )
    return SplitFrames(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
    )


def group_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> SplitFrames:
    groups = df[group_col].values
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(outer.split(df, groups=groups))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    rel_val_size = val_size / (1.0 - test_size)
    inner = GroupShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=seed)
    inner_groups = train_val_df[group_col].values
    train_idx, val_idx = next(inner.split(train_val_df, groups=inner_groups))
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    return SplitFrames(train_df=train_df, val_df=val_df, test_df=test_df)


def extrap_jet_split(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    test_size: float,
    val_size: float,
    seed: int,
    extrap_group: str = "JET",
) -> SplitFrames:
    del seed

    if len(df) < 3:
        raise ValueError("Need at least 3 rows to build the extrap_jet split.")

    ranked = df.sort_values(target_col, ascending=False, kind="mergesort").reset_index(drop=True)
    top_n = max(1, min(int(np.ceil(test_size * len(ranked))), len(ranked) - 1))
    top_bucket = ranked.iloc[:top_n].copy()
    remain_df = ranked.iloc[top_n:].copy()

    test_mask = top_bucket[group_col].astype(str) == extrap_group
    test_df = top_bucket.loc[test_mask].copy()
    if test_df.empty:
        raise ValueError(
            f"Top-{test_size:.0%} target bucket contains no rows from group {extrap_group!r}."
        )

    train_val_pool = pd.concat([remain_df, top_bucket.loc[~test_mask]], axis=0, ignore_index=True)
    val_n = max(1, min(int(np.ceil(val_size * len(ranked))), len(train_val_pool) - 1))
    train_df, val_df = _split_by_top_target(train_val_pool, target_col=target_col, holdout_size=val_n)
    return SplitFrames(train_df=train_df, val_df=val_df, test_df=test_df.reset_index(drop=True))


def refit_train_val_split(
    train_val_df: pd.DataFrame,
    split_type: str,
    group_col: str,
    target_col: str,
    seed: int,
    val_fraction: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_type == "random":
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_fraction,
            random_state=seed,
            shuffle=True,
        )
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    if split_type == "group":
        gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        groups = train_val_df[group_col].values
        train_idx, val_idx = next(gss.split(train_val_df, groups=groups))
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    if split_type == "extrap_jet":
        val_n = int(np.ceil(val_fraction * len(train_val_df)))
        return _split_by_top_target(train_val_df, target_col=target_col, holdout_size=val_n)

    raise ValueError(f"Unsupported split_type: {split_type}")


def build_split(
    df: pd.DataFrame,
    split_type: str,
    group_col: str,
    target_col: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> SplitFrames:
    if split_type == "random":
        return random_split(df=df, test_size=test_size, val_size=val_size, seed=seed)
    if split_type == "group":
        return group_split(
            df=df,
            group_col=group_col,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
        )
    if split_type == "extrap_jet":
        return extrap_jet_split(
            df=df,
            group_col=group_col,
            target_col=target_col,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
        )
    raise ValueError(f"Unsupported split_type: {split_type}")
