from imblearn.base import SamplerMixin
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y
import pandas as pd, numpy as np
import smogn


class SMOGNSampler(BaseEstimator, SamplerMixin):
    """
    Regression oversampler that wraps `smogn.smoter()` so it can be placed
    inside an `imblearn.Pipeline`.

    Parameters
    ----------
    y_col : str, default="target"
        Temporary name for the target column `smogn` expects.
    k : int, default=5
        Number of nearest neighbours.
    samp_method : {"balance", "extreme"}, default="extreme"
        SMOGN sampling mode.
    over_n, under_n : int, default (200, 100)
        Over‑ and under‑sampling percentages.
    **kwargs : dict
        Any extra key‑word args accepted by `smogn.smoter()`.
    """
    # ---- new: satisfy scikit‑learn 1.4+ clone/validation ----------
    _parameter_constraints: dict = {}    

    def __init__(self, y_col="target", k=5,
                 samp_method="extreme", over_n=200, under_n=100, **kwargs):
        self.y_col = y_col
        self.k = k
        self.samp_method = samp_method
        self.over_n = over_n
        self.under_n = under_n
        self.kwargs = kwargs

    # ----------------------------------------------------------------
    def _fit(self, X, y):
        # Samplers do not learn parameters; just return self
        return self

    def _fit_resample(self, X, y):
        # Basic consistency check; no classification target validation
        X, y = check_X_y(X, y, accept_sparse=["csr", "csc"])

        # Build DataFrame because `smogn` requires one
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])

        df[self.y_col] = y

        df_res = smogn.smoter(
            data=df,
            y=self.y_col,
            k=self.k,
            samp_method=self.samp_method,
            over_n=self.over_n,
            under_n=self.under_n,
            **self.kwargs,
        )

        y_res = df_res[self.y_col].to_numpy()
        X_res = df_res.drop(columns=[self.y_col]).to_numpy()

        return X_res, y_res
