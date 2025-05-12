# torch_estimator.py
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import f1_score, balanced_accuracy_score

def f1_from_soft_labels(y_true, y_pred):
    y_pred_bin = (y_pred >= 0.5).astype(int)
    y_true = (np.asarray(y_true) >= 0.5).astype(int)
    return f1_score(y_true, y_pred_bin)

class TorchSoftRegressor(BaseEstimator, RegressorMixin):
    """
    Minimal sklearn‑style wrapper around a PyTorch MLP that predicts
    a single logit for soft‑label BCE training.
    """
    def __init__(
        self,
        in_features=27,
        lr=1e-3,
        max_epochs=20,
        device='cpu',
        batch_size=32,
        verbose=0,
        random_state=42,
        patience=10
    ):
        self.in_features = in_features
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        self.patience = patience

    # ------------------------------------------------------------------
    def _build_net(self):
        torch.manual_seed(self.random_state)
        
        net = nn.Sequential(
            nn.Linear(self.in_features, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Linear(64, 100),
            nn.BatchNorm1d(100),
            nn.GELU(),

            nn.Linear(100, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1), # Best Tested Dropout so far: Dropout = 0.1

            nn.Linear(128, 148),
            nn.BatchNorm1d(148),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(148, 190),
            nn.BatchNorm1d(190),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(190, 270),
            nn.BatchNorm1d(270),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(270, 250),
            nn.BatchNorm1d(250),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(250, 180),
            nn.BatchNorm1d(180),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(180, 120),
            nn.BatchNorm1d(120),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(120, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1)
        )
        return net.to(self.device)

    # fit() – train with BCEWithLogitsLoss on float32 data
    def fit(self, X, y):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device).view(-1, 1)

        self.net_ = self._build_net()
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        y_hard = (y >= 0.5).float() 
        pos_frac = y_hard.mean()
        pos_weight = torch.tensor([(1 - pos_frac) / pos_frac], device=self.device)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        val_frac = 0.1
        n_val = int(len(X) * val_frac)
        perm  = torch.randperm(len(X), device=self.device)
        idx_val, idx_tr = perm[:n_val], perm[n_val:]

        X_val_t, y_val_t = X[idx_val], y[idx_val]
        X_tr_t, y_tr_t = X[idx_tr], y[idx_tr]

        train_ds = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
        loader = torch.utils.data.DataLoader(train_ds,
                                             batch_size=self.batch_size,
                                             shuffle=True)

        # 1) initialise bookkeeping -------------------------------------
        best_bce   = float('inf')
        best_state = None
        patience   = 5
        since_improve = 0

        # 2) training epochs --------------------------------------------
        for epoch in range(self.max_epochs):
            self.net_.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = crit(self.net_(xb), yb)
                loss.backward()
                opt.step()

            # ---- validation pass ---------------------------------------
            self.net_.eval()
            with torch.no_grad():
                val_logits = self.net_(X_val_t)
                val_loss   = crit(val_logits, y_val_t).item()
                val_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()
                y_val_hard = (y_val_t >= 0.5).cpu().numpy().ravel()
                val_f1  = f1_score(y_val_hard, (val_probs >= 0.5))
                acc  = balanced_accuracy_score(y_val_hard, (val_probs >= 0.5))

            if self.verbose:
                print(f"epoch {epoch:02d}  train_loss {loss.item():.4f} "
                      f"val_BCE {val_loss:.4f}  F1 {val_f1:.4f}  Balanced Acc {acc:.4f}")

            # ---- early‑stopping logic ----------------------------------
            if val_loss < best_bce - 1e-4:      # significant improvement
                best_bce = val_loss
                since_improve = 0
                # clone the entire state dict (tensor.clone ensures no gradual overwrite)
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                since_improve += 1
                if since_improve > patience:
                    if self.verbose:
                        print("Early stopping triggered.")
                    break

        # 3) restore best weights ---------------------------------------
        if best_state is not None:
            self.net_.load_state_dict(best_state)
            
        return self

    # predict raw logits
    def predict(self, X):
        self.net_.eval()
        with torch.no_grad():
            X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            logits = self.net_(X).cpu().numpy().squeeze()
        return logits  # caller can sigmoid if needed

    # sklearn will clone() via get_params / set_params
    def get_params(self, deep=True):
        return {k: v for k, v in self.__dict__.items()
                if not k.endswith('_')}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
