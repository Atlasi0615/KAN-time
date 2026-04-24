
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@dataclass
class LOTOFold:
    held_out_group: str
    train_dev_df: pd.DataFrame
    test_df: pd.DataFrame


def iter_loto_folds(df: pd.DataFrame, group_col: str) -> Iterator[LOTOFold]:
    groups: List[str] = sorted(pd.Series(df[group_col]).astype(str).unique().tolist())
    for held_out in groups:
        test_mask = df[group_col].astype(str) == held_out
        test_df = df.loc[test_mask].reset_index(drop=True)
        train_dev_df = df.loc[~test_mask].reset_index(drop=True)
        yield LOTOFold(
            held_out_group=held_out,
            train_dev_df=train_dev_df,
            test_df=test_df,
        )


def group_train_val_split(
    train_dev_df: pd.DataFrame,
    group_col: str,
    val_size: float,
    seed: int,
):
    unique_groups = pd.Series(train_dev_df[group_col]).astype(str).nunique()
    if unique_groups < 2:
        raise ValueError(
            f"Need at least 2 groups in train_dev_df for grouped validation, got {unique_groups}."
        )
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    groups = train_dev_df[group_col].astype(str).values
    train_idx, val_idx = next(gss.split(train_dev_df, groups=groups))
    train_df = train_dev_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_dev_df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df
