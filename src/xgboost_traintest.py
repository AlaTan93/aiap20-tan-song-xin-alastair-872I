import joblib
import xgboost as xgb
import numpy as np, pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

def train_and_save(X_train, X_val, y_train, y_val, location="saved/xgboost.pkl"):
    numeric_cols = ['Age', 'Campaign Calls', 'Previous Contact Days']
    
    oversampler = SMOTE(random_state=0)

    model = xgb.XGBClassifier(objective="binary:logistic",
                                      eval_metric="logloss",
                                      max_depth=4,
                                      learning_rate=0.05,
                                      n_estimators=600,
                                      subsample=0.8,
                                      colsample_bytree=0.8,
                                      random_state=0,
                                      n_jobs=-1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), numeric_cols)],
        remainder="passthrough"
    )
    
    pipe = Pipeline(steps=[
        ('prep', preprocessor),
        ("oversample", oversampler),
        ("model", model)])
    
    param_grid = {
        "model__n_estimators": [300, 600, 900],
        "model__max_depth":    [3, 4, 6, None],
        "model__learning_rate":[0.03, 0.05, 0.1],
        "model__subsample":    [0.6, 0.8, 1.0],
        "model__colsample_bytree":[0.6, 0.8, 1.0],
    }
    search = GridSearchCV(pipe, param_grid,
                          scoring="f1",
                          cv=cv, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    
    y_pred = search.best_estimator_.predict(X_val)
    print(search.best_params_)
    print("Hold‑out F1:", f1_score(y_val, y_pred))
        
    joblib.dump(search.best_estimator_, location)
    print("Saved →", location)

    best_pipe = search.best_estimator_
    
    # Get the fitted StandardScaler from inside the ColumnTransformer
    scaler = best_pipe.named_steps["prep"].named_transformers_["num"]

    # Save the scaler
    joblib.dump(scaler, "saved/xgboost_scaler.pkl")

def test_model():
    pass
