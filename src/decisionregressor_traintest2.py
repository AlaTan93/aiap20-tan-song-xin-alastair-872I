from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor          
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, log_loss
import smote_with_soft as smote
import numpy as np
import joblib

# ── metrics -----------------------------------------------------------------
def bce_score(y_true, y_prob, eps=1e-6):
    """Binary cross‑entropy on soft labels (higher is better)."""
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -log_loss(y_true, y_prob)

bce_scorer = make_scorer(bce_score, greater_is_better=True, needs_proba=False)

# ── training function -------------------------------------------------------
def train_and_save_soft(
        X_train, X_val, y_train_soft, y_val_soft,
        pipe_path="saved/dtr_soft_pipe.pkl",
        scaler_path="saved/dtr_soft_scaler.pkl"):
    num_cols = ["Age", "Campaign Calls", "Previous Contact Days"]

    pre = ColumnTransformer(
        [("num", StandardScaler(), num_cols)],
        remainder="passthrough"
    )

    sampler = smote.SMOTEWithSoftLabels(random_state=0)

    model = DecisionTreeRegressor(random_state=0)

    pipe = Pipeline([
        ("prep",   pre),
        ("smote",  sampler),        # uses the wrapper above
        ("model",  model)
    ])

    param_grid = {"model__max_depth": [3, 5, 7, None]}

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=bce_scorer,         
        cv=KFold(5, shuffle=True, random_state=0),
        refit=True,
        n_jobs=-1
    )
    search.fit(X_train, y_train_soft)

    # ── hold‑out evaluation ───────────────────────────────────────────
    y_pred_soft = search.best_estimator_.predict(X_val)
    print("Best params :", search.best_params_)
    print("Hold‑out BCE:", bce_score(y_val_soft, y_pred_soft))

    # ── save artefacts ────────────────────────────────────────────────
    joblib.dump(search.best_estimator_, pipe_path)
    print("Saved pipeline →", pipe_path)

    scaler = search.best_estimator_.named_steps["prep"].named_transformers_["num"]
    joblib.dump(scaler, scaler_path)
    print("Saved scaler   →", scaler_path)
