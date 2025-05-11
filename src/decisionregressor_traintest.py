import joblib
import numpy as np, pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss, make_scorer, f1_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

def train_and_save(X_train, X_val, y_train, y_val, location="saved/decisionregressor.pkl"):
    numeric_cols = ['Age', 'Campaign Calls', 'Previous Contact Days']

    scaler = StandardScaler()
    model = DecisionTreeRegressor(max_depth=3, random_state=0)

    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), numeric_cols)],
        remainder="passthrough"
    )

    pipe = Pipeline(steps=[
        ('prep', preprocessor),
        ("model", model)])

    param_grid = {
        "model__max_depth": [3, 5, 6, 7, 8, None],
        "model__min_samples_split": [2, 5, 10, 20]
    }

    bce_scorer = make_scorer(bce_score, greater_is_better=True)    
    
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring=bce_scorer,
        cv=5,
        refit=True
    )
    search.fit(X_train, y_train)
    
    y_pred = search.best_estimator_.predict(X_val)
    print(search.best_params_)
    print("Hold‑out BCE:", bce_score(y_val, y_pred))
    print("Hold‑out F1:", f1_from_soft_labels(y_val, y_pred))
        
    joblib.dump(search.best_estimator_, location)
    print("Saved →", location)

    best_pipe = search.best_estimator_
    
    # Get the fitted StandardScaler from inside the ColumnTransformer
    scaler = best_pipe.named_steps["prep"].named_transformers_["num"]

    # Save the scaler
    joblib.dump(scaler, "saved/decision_scaler.pkl")

def bce_score(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
    y_true = (np.asarray(y_true) >= 0.5).astype(int)
    return -log_loss(y_true, y_pred)

def f1_from_soft_labels(y_true, y_pred):
    y_pred_bin = (y_pred >= 0.5).astype(int)
    y_true = (np.asarray(y_true) >= 0.5).astype(int)
    return f1_score(y_true, y_pred_bin)


