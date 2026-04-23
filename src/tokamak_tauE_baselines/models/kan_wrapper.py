from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Dict, List, Sequence

import numpy as np
import torch

KAN = None
_KAN_IMPORT_ERROR = None
_KAN_IMPORT_SOURCE = None


def _load_kan_class():
    global KAN, _KAN_IMPORT_ERROR, _KAN_IMPORT_SOURCE
    if KAN is not None:
        return KAN

    candidates = [("kan", "KAN"), ("pykan", "KAN")]
    last_error = None
    for module_name, attr_name in candidates:
        try:
            module = import_module(module_name)
            cls = getattr(module, attr_name)
            KAN = cls
            _KAN_IMPORT_SOURCE = getattr(module, "__file__", module_name)
            _KAN_IMPORT_ERROR = None
            return KAN
        except Exception as exc:  # pragma: no cover
            last_error = exc

    _KAN_IMPORT_ERROR = last_error
    return None


@dataclass
class KANTrainOutput:
    model: object
    history: Dict[str, List[float]]


def ensure_kan_installed() -> None:
    if _load_kan_class() is None:
        detail = (
            f"Original import error: {type(_KAN_IMPORT_ERROR).__name__}: {_KAN_IMPORT_ERROR}"
            if _KAN_IMPORT_ERROR
            else "No additional import details available."
        )
        raise ImportError(
            "Failed to import the KAN class. This is usually one of:\n"
            "1) the active interpreter is not the one where `pykan` was installed;\n"
            "2) another package/module named `kan` is shadowing `pykan`;\n"
            "3) `pykan` imported but one of its own dependencies failed.\n\n"
            "Try these checks:\n"
            "  python -c \"import sys; print(sys.executable)\"\n"
            "  python -m pip show pykan\n"
            "  python -c \"import kan; print(kan.__file__); from kan import KAN; print(KAN)\"\n\n"
            + detail
        )


def train_kan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: Sequence[int],
    grid: int,
    k: int,
    adam_steps: int,
    adam_lr: float,
    lbfgs_steps: int,
    lamb: float,
    lamb_entropy: float,
    update_grid: bool,
    grid_update_num: int,
    start_grid_update_step: int,
    stop_grid_update_step: int,
    batch: int,
    seed: int,
    device: str,
) -> KANTrainOutput:
    ensure_kan_installed()

    width = [X_train.shape[1]] + list(hidden_dims) + [1]
    model = KAN(width=width, grid=grid, k=k, seed=seed, device=device)

    if hasattr(model, "speed"):
        model.speed()

    dataset = {
        "train_input": torch.tensor(X_train, dtype=torch.float32, device=device),
        "train_label": torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device),
        "test_input": torch.tensor(X_val, dtype=torch.float32, device=device),
        "test_label": torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32, device=device),
    }

    history: Dict[str, List[float]] = {}

    if adam_steps > 0:
        hist_adam = model.fit(
            dataset,
            opt="Adam",
            steps=adam_steps,
            lr=adam_lr,
            lamb=lamb,
            lamb_entropy=lamb_entropy,
            update_grid=update_grid,
            grid_update_num=grid_update_num,
            start_grid_update_step=start_grid_update_step,
            stop_grid_update_step=stop_grid_update_step,
            batch=batch,
            log=max(1, adam_steps + 1),
        )
        history["adam"] = _history_to_dict(hist_adam)

    if lbfgs_steps > 0:
        hist_lbfgs = model.fit(
            dataset,
            opt="LBFGS",
            steps=lbfgs_steps,
            lamb=lamb,
            lamb_entropy=lamb_entropy,
            update_grid=False,
            batch=batch,
            log=max(1, lbfgs_steps + 1),
        )
        history["lbfgs"] = _history_to_dict(hist_lbfgs)

    return KANTrainOutput(model=model, history=history)


def predict_kan(model: object, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(Xt)
    if isinstance(pred, tuple):
        pred = pred[0]
    return pred.detach().cpu().numpy().reshape(-1)


def try_save_kan_state(model: object, path: str) -> None:
    try:
        torch.save(model.state_dict(), path)
    except Exception:
        pass


def _history_to_dict(hist: object):
    return _to_python(hist)


def _to_python(obj: object):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_python(v) for v in obj]
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        try:
            return obj.detach().cpu().tolist()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)