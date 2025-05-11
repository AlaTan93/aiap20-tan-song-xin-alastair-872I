import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import TensorDataset, DataLoader

class TabularNet(nn.Module):
    """
    Simple fully‑connected network for 27‑dim tabular data.
    Last layer returns *one* logit; apply sigmoid only for metrics.
    """
    def __init__(self, in_features: int = 27):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),

            nn.Linear(64, 1)      # logits (no activation!)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)   # shape (batch,)

def train_epoch(model, loader, optimizer, device="cuda"):
    model.train()
    epoch_loss = 0
    for X, y in loader:                     # y ∈ [0,1] floats
        X, y = X.to(device), y.to(device).float()

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X.size(0)

    return epoch_loss / len(loader.dataset)

@torch.inference_mode()
def evaluate(model, loader, device="cuda"):
    model.eval()
    all_probs, all_targets = [], []

    for X, y in loader:
        X = X.to(device)
        probs = torch.sigmoid(model(X)).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.numpy())

    y_true  = np.concatenate(all_targets)
    y_prob  = np.concatenate(all_probs)

    return {
        "BCE":  criterion(
                    torch.from_numpy(
                        np.log(y_prob / (1 - y_prob + 1e-7) + 1e-7)  # logits again
                    ), torch.from_numpy(y_true)
                ).item(),
        "ROC‑AUC": roc_auc_score(y_true, y_prob),
        "PR‑AUC":  average_precision_score(y_true, y_prob),
    }

def train_and_save(X_train, X_val, y_

X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32)

train_ds  = TensorDataset(X_train_t, y_train_t)
train_dl  = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = TabularNet().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

best_auc, best_state = 0.0, None
for epoch in range(50):
    train_loss = train_epoch(model, train_dl, optimizer, device)
    metrics    = evaluate(model, val_dl, device)
    scheduler.step(metrics["ROC‑AUC"])

    if metrics["ROC‑AUC"] > best_auc:
        best_auc, best_state = metrics["ROC‑AUC"], model.state_dict().copy()

    print(f"Epoch {epoch:02d} │ loss {train_loss:.4f} │ "
          f"val PR‑AUC {metrics['PR‑AUC']:.4f}")

model.load_state_dict(best_state)   # restore best epoch
torch.save(model.state_dict(), "subscription_net.pt")

