from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .splits import SplitFrames


@dataclass
class PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    metadata_train: pd.DataFrame
    metadata_val: pd.DataFrame
    metadata_test: pd.DataFrame
    x_scaler: StandardScaler | None
    y_scaler: StandardScaler | None
    log_target: bool

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).reshape(-1, 1)
        if self.y_scaler is not None:
            y = self.y_scaler.inverse_transform(y)
        return y.reshape(-1)

    def true_test_log(self) -> np.ndarray:
        return self.inverse_transform_y(self.y_test)


def load_dataframe(
    csv_path: str | Path,
    features: Sequence[str],
    target: str,
    metadata_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    metadata_cols = list(metadata_cols or [])
    required_cols = list(features) + [target] + metadata_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    use_cols = list(dict.fromkeys(required_cols))
    df = df[use_cols].copy()

    # Drop rows with missing values in required columns.
    df = df.dropna(subset=list(features) + [target]).reset_index(drop=True)

    # Enforce positivity for log transforms.
    pos_mask = np.ones(len(df), dtype=bool)
    for col in list(features) + [target]:
        pos_mask &= df[col].astype(float).to_numpy() > 0.0
    df = df.loc[pos_mask].reset_index(drop=True)

    return df


def log_feature_names(features: Sequence[str]) -> List[str]:
    return [f"log_{f}" for f in features]


def _build_matrix(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    log_inputs: bool,
    log_target: bool,
) -> tuple[np.ndarray, np.ndarray]:
    X = df[list(features)].astype(float).to_numpy()
    y = df[[target]].astype(float).to_numpy()

    if log_inputs:
        X = np.log(X)
    if log_target:
        y = np.log(y)

    return X, y


def prepare_data(
    split_frames: SplitFrames,
    features: Sequence[str],
    target: str,
    metadata_cols: Sequence[str],
    log_inputs: bool,
    log_target: bool,
    scale_x: bool,
    scale_y: bool,
) -> PreparedData:
    X_train, y_train = _build_matrix(split_frames.train_df, features, target, log_inputs, log_target)
    X_val, y_val = _build_matrix(split_frames.val_df, features, target, log_inputs, log_target)
    X_test, y_test = _build_matrix(split_frames.test_df, features, target, log_inputs, log_target)

    x_scaler = StandardScaler() if scale_x else None
    y_scaler = StandardScaler() if scale_y else None

    if x_scaler is not None:
        X_train = x_scaler.fit_transform(X_train)
        X_val = x_scaler.transform(X_val)
        X_test = x_scaler.transform(X_test)

    if y_scaler is not None:
        y_train = y_scaler.fit_transform(y_train)
        y_val = y_scaler.transform(y_val)
        y_test = y_scaler.transform(y_test)

    feature_names = log_feature_names(features) if log_inputs else list(features)

    return PreparedData(
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.float32).reshape(-1),
        X_val=X_val.astype(np.float32),
        y_val=y_val.astype(np.float32).reshape(-1),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.float32).reshape(-1),
        feature_names=feature_names,
        metadata_train=split_frames.train_df[list(metadata_cols)].copy(),
        metadata_val=split_frames.val_df[list(metadata_cols)].copy(),
        metadata_test=split_frames.test_df[list(metadata_cols)].copy(),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        log_target=log_target,
    )


def combine_train_val(split_frames: SplitFrames) -> pd.DataFrame:
    return pd.concat([split_frames.train_df, split_frames.val_df], axis=0).reset_index(drop=True)
