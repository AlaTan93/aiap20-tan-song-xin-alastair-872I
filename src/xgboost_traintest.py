import joblib
import xgboost as xgb
import numpy as np, pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
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
    print("Hold‑out Balanced Accuracy:", balanced_accuracy_score((y_val >= 0.5), (y_pred >= 0.5)))
        
    joblib.dump(search.best_estimator_, location)
    print("Saved →", location)

    best_pipe = search.best_estimator_

def f1_from_soft_labels(y_true, y_pred):
    y_pred_bin = (y_pred >= 0.5).astype(int)
    y_true = (np.asarray(y_true) >= 0.5).astype(int)
    return f1_score(y_true, y_pred_bin)

def load_and_test(X, y, location="saved/xgboost.pkl"):
    pipe = joblib.load(location)

    y_pred = pipe.predict(X)
    print("F1:", f1_from_soft_labels(y, y_pred))
    print("Balanced Accuracy:", balanced_accuracy_score((y >= 0.5), (y_pred >= 0.5)))
