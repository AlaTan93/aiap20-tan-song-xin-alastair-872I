from torch_estimator import TorchSoftRegressor
import torch, joblib
import numpy as np
from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import log_loss, make_scorer, f1_score, balanced_accuracy_score

def to_soft(X, y):
    return X, y.astype(np.float32) * 0.85 + 0.05

def to_float32(X):
    return X.astype(np.float32)

def bce_score(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
    y_true = (np.asarray(y_true) >= 0.5).astype(int)
    return log_loss(y_true, y_pred)

def f1_from_soft_labels(y_true, y_pred):
    y_pred_bin = (y_pred >= 0.5).astype(int)
    y_true = (np.asarray(y_true) >= 0.5).astype(int)
    return f1_score(y_true, y_pred_bin)

def cross_val_with_soft_labels(X, y, pipeline, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    best_score = float('inf')
    best_model = None

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Clone the pipeline to avoid data leakage between folds
        pipe = clone(pipeline)

        # Fit on train
        pipe.fit(X_train, y_train)

        # Predict on test
        y_pred = torch.sigmoid(torch.tensor(pipe.predict(X_test))).numpy()

        # Score with raw labels
        score = bce_score(y_test, y_pred)
        f1_score = f1_from_soft_labels(y_test, y_pred)
        acc = balanced_accuracy_score((y_test >= 0.5), (y_pred >= 0.5))
        print(f"  Hard BCE: {score:.4f}")
        print(f"  F1: {f1_score:.4f}")
        print(f"  Balanced Acc: {acc:.4f}")
        scores.append(score)

        if score < best_score:
            best_score = score
            best_model = pipe

    print(f"\nMean BCE: {np.mean(scores):.4f}")
    return best_model

def load_and_test(X, y, location = "saved/best_deep_pipeline.pkl"):
    pipe = joblib.load(location)

    y_pred = pipe.predict(X)
    print("BCE:", bce_score(y, y_pred))
    print("F1:", f1_from_soft_labels(y, y_pred))
    print("Balanced Accuracy:", balanced_accuracy_score((y >= 0.5), (y_pred >= 0.5)))

def train_and_save(X_train, X_val, y_train, y_val):
    soft = FunctionSampler(func=to_soft, validate=False)
    tof32 = FunctionTransformer(func=to_float32, validate=False)    

    num_cols = ["Age", "Campaign Calls", "Previous Contact Days"]
    prep = ColumnTransformer([("num", StandardScaler(), num_cols)],
                             remainder="passthrough")

    reg = TorchSoftRegressor(
        in_features=27,
        lr=1e-6,
        max_epochs=100,
        batch_size=900, # 512 best option
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=False
    )

    pipe = Pipeline([
        ("prep",  prep),
        ("cast",  tof32),               # X float32    
        ("smote", SMOTE(random_state=0)),
        ("soft",  soft),                
        ("model", reg)
    ])

    # Get the best model after cross-validation
    best_pipe = cross_val_with_soft_labels(X_train, y_train, pipe)

    # Evaluate on validation set
    y_val_pred = torch.sigmoid(torch.tensor(best_pipe.predict(X_val))).numpy()
    print("Deep Holdout Hard BCE:", bce_score(y_val, y_val_pred))
    print("Holdout F1:", f1_from_soft_labels(y_val, y_val_pred))
    print("Holdout Acc:", balanced_accuracy_score((y_val >= 0.5), (y_val_pred >= 0.5)))

    location = "saved/best_deep_pipeline.pkl"
    joblib.dump(best_pipe, location)
    print("Saved →", location)

