from imblearn.over_sampling import SMOTE
from sklearn.utils import check_X_y
import numpy as np

class SMOTEWithSoftLabels(SMOTE):
    """
    Use SMOTE on *hard* labels, then map back to soft 0.0 / 1.0.
    Keeps the imblearn API, but bypasses the classification‑target check.
    """
    def __init__(self, threshold=0.5, **smote_kwargs):
        super().__init__(**smote_kwargs)
        self.threshold = threshold

    # <-- override the full method
    def fit_resample(self, X, y_soft):
        # basic input validation (no class check)
        X, y_soft = check_X_y(X, y_soft, accept_sparse=["csr", "csc"])

        # hard‑threshold so SMOTE can work
        y_hard = (y_soft >= self.threshold).astype(int)

        # call SMOTE's own fit_resample with the *hard* labels
        X_res, y_hard_res = super().fit_resample(X, y_hard)

        # convert resampled labels back to float 0.0 / 1.0
        y_soft_res = y_hard_res.astype(float)

        return X_res, y_soft_res
