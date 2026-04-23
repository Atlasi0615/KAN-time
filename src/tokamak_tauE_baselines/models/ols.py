from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class OLSResult:
    model: sm.regression.linear_model.RegressionResultsWrapper
    predictions: np.ndarray

    def coefficient_frame(self) -> pd.DataFrame:
        params = self.model.params
        conf = self.model.conf_int()
        pvalues = self.model.pvalues
        out = pd.DataFrame(
            {
                "term": params.index,
                "coef": params.values,
                "ci_low": conf[0].values,
                "ci_high": conf[1].values,
                "p_value": pvalues.values,
            }
        )
        return out


def fit_ols(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    add_constant: bool = True,
) -> OLSResult:
    Xtr = X_train.copy()
    Xte = X_test.copy()
    if add_constant:
        Xtr = sm.add_constant(Xtr, has_constant="add")
        Xte = sm.add_constant(Xte, has_constant="add")

    model = sm.OLS(y_train.reshape(-1), Xtr).fit()
    preds = model.predict(Xte)
    return OLSResult(model=model, predictions=np.asarray(preds).reshape(-1))
