from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass
class SplitFrames:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


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


def build_split(
    df: pd.DataFrame,
    split_type: str,
    group_col: str,
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
    raise ValueError(f"Unsupported split_type: {split_type}")
