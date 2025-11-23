import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import metl
import numpy as np

# ============================================================
# Small Spearman helper (no SciPy needed)
# ============================================================

def simple_spearman(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    def rank(a):
        order = np.argsort(a)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(a))
        vals, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, r)
        avg = sums / counts
        return avg[inv]

    r1 = rank(y_true)
    r2 = rank(y_pred)
    if r1.std() == 0 or r2.std() == 0:
        return 0.0
    return float(np.corrcoef(r1, r2)[0, 1])


# ============================================================
# Config
# ============================================================

BATCH_SIZE = 256

EPOCHS_HEAD = 15       # Phase 1: head-only
EPOCHS_FINETUNE = 5    # Phase 2: partial unfreeze
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4

VAL_FRAC = 0.1         # 10% of non-elites as validation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DMS_PATH = os.path.join(
    BASE_DIR, "metl-pub", "data", "dms_data", "tem-1", "tem-1.tsv"
)

# WT TEM-1 sequence (same as used before)
WT_TEM1 = (
    "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRP"
    "EERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVREL"
    "CSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTM"
    "PAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGS"
    "RGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
).replace(" ", "")


# ============================================================
# 1. Load and normalize TEM-1 DMS data
# ============================================================

df = pd.read_csv(DMS_PATH, sep="\t")  # ['variant', 'num_mutations', 'score']

mean = df["score"].mean()
std = df["score"].std()
df["score_norm"] = (df["score"] - mean) / std

# Sort by score descending
df_sorted = df.sort_values("score", ascending=False).reset_index(drop=True)

# ------------------------------------------------------------
# Define structured top-30 holdout:
# take indices 0-4, 10-14, 20-24, 30-34, 40-44, 50-54  (6 blocks * 5 = 30)
# ------------------------------------------------------------
elite_indices = []
start = 0
while len(elite_indices) < 30:
    take = list(range(start, start + 5))
    elite_indices.extend(take)
    start += 10

elite_indices = elite_indices[:30]  # just in case
heldout_df = df_sorted.iloc[elite_indices].copy()

# Non-elites = all others
mask = np.ones(len(df_sorted), dtype=bool)
mask[elite_indices] = False
non_elite_df = df_sorted[mask].copy()

print(f"Total variants: {len(df)}")
print("Held out elites (n=30):")
print(heldout_df[["variant", "score"]].head(10))

# Shuffle non-elites for train/val split
non_elite_df = non_elite_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

val_size = int(len(non_elite_df) * VAL_FRAC)
val_df = non_elite_df.iloc[:val_size].copy()
train_df = non_elite_df.iloc[val_size:].copy()

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Held-out elites: {len(heldout_df)}")


# ============================================================
# 2. Load TEM-1 local source model (PREhfC22)
# ============================================================

source_model, data_encoder = metl.get_from_uuid("PREhfC22")
source_model.to(DEVICE)
source_model.eval()

# Freeze all params initially; we control unfreezing later
for p in source_model.parameters():
    p.requires_grad = False


# ============================================================
# 3. Dataset
# ============================================================

class Tem1DmsDataset(Dataset):
    def __init__(self, df_sub, wt_seq, encoder):
        self.variants = df_sub["variant"].tolist()
        self.labels = df_sub["score_norm"].astype(float).values

        encoded = encoder.encode_variants(wt_seq, self.variants)
        if isinstance(encoded, dict):
            self.encoded = encoded["encoded_seqs"]
        else:
            self.encoded = encoded

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


train_dataset   = Tem1DmsDataset(train_df,   WT_TEM1, data_encoder)
val_dataset     = Tem1DmsDataset(val_df,     WT_TEM1, data_encoder)
heldout_dataset = Tem1DmsDataset(heldout_df, WT_TEM1, data_encoder)
all_dataset     = Tem1DmsDataset(df,         WT_TEM1, data_encoder)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
all_loader   = DataLoader(all_dataset,   batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# 4. Model: source backbone (55-dim) + nonlinear head
# ============================================================

class FtModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model        # outputs [batch, 55]
        self.head = nn.Sequential(
            nn.Linear(55, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        feats = self.base(x)          # [batch, 55]
        out = self.head(feats)        # [batch, 1]
        return out.squeeze(-1)        # [batch]

ft_model = FtModel(source_model).to(DEVICE)

# Weighted MSE: emphasize higher-fitness (score_norm > 0)
def weighted_mse(preds, target):
    w = 1.0 + 0.5 * torch.clamp(target, min=0.0)
    return (w * (preds - target) ** 2).mean()


# ============================================================
# 5. Phase 1: train ONLY the new head
# ============================================================

optimizer = torch.optim.Adam(
    ft_model.head.parameters(),
    lr=LR_HEAD,
)

best_val = float("inf")
best_state = None

for epoch in range(1, EPOCHS_HEAD + 1):
    # train
    ft_model.train()
    total_train = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()
        preds = ft_model(batch_x)
        loss = weighted_mse(preds, batch_y)
        loss.backward()
        optimizer.step()

        total_train += loss.item() * batch_x.size(0)
    avg_train = total_train / len(train_dataset)

    # validate
    ft_model.eval()
    total_val = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            preds = ft_model(batch_x)
            loss = weighted_mse(preds, batch_y)
            total_val += loss.item() * batch_x.size(0)
    avg_val = total_val / len(val_dataset)

    print(f"[Head] Epoch {epoch}/{EPOCHS_HEAD} - "
          f"train MSE: {avg_train:.4f} - val MSE: {avg_val:.4f}")

    if avg_val < best_val:
        best_val = avg_val
        best_state = {k: v.cpu() for k, v in ft_model.state_dict().items()}

# restore best head-only weights
if best_state is not None:
    ft_model.load_state_dict(best_state)
    ft_model.to(DEVICE)
    print(f"Loaded best head-only model (val MSE={best_val:.4f})")


# ============================================================
# 6. Phase 2: partial unfreeze for light fine-tuning
# ============================================================

# Unfreeze a subset of source_model (e.g. fc1/prediction-like layers)
for name, p in source_model.named_parameters():
    if ("fc1" in name) or ("prediction" in name):
        p.requires_grad = True

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, ft_model.parameters()),
    lr=LR_FINETUNE,
)

best_val_ft = best_val
best_state_ft = {k: v.cpu() for k, v in ft_model.state_dict().items()}

for epoch in range(1, EPOCHS_FINETUNE + 1):
    ft_model.train()
    total_train = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()
        preds = ft_model(batch_x)
        loss = weighted_mse(preds, batch_y)
        loss.backward()
        optimizer.step()

        total_train += loss.item() * batch_x.size(0)
    avg_train = total_train / len(train_dataset)

    # validate
    ft_model.eval()
    total_val = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            preds = ft_model(batch_x)
            loss = weighted_mse(preds, batch_y)
            total_val += loss.item() * batch_x.size(0)
    avg_val = total_val / len(val_dataset)

    print(f"[FT] Epoch {epoch}/{EPOCHS_FINETUNE} - "
          f"train MSE: {avg_train:.4f} - val MSE: {avg_val:.4f}")

    if avg_val < best_val_ft:
        best_val_ft = avg_val
        best_state_ft = {k: v.cpu() for k, v in ft_model.state_dict().items()}

# restore best model
ft_model.load_state_dict(best_state_ft)
ft_model.to(DEVICE)
print(f"Loaded best fine-tuned model (val MSE={best_val_ft:.4f})")


# ============================================================
# 7. Evaluate on held-out elites (top-30)
# ============================================================

ft_model.eval()

heldout_encoded = torch.tensor(heldout_dataset.encoded, dtype=torch.long).to(DEVICE)
with torch.no_grad():
    heldout_pred_norm = ft_model(heldout_encoded).cpu().numpy()

heldout_pred = heldout_pred_norm * std + mean
heldout_df["pred"] = heldout_pred

print("\nHeld-out elites (truth vs prediction):")
print(heldout_df[["variant", "score", "pred"]].sort_values("score", ascending=False))

# Spearman only on the 30 held-out elites
sp_elites = simple_spearman(heldout_df["score"], heldout_df["pred"])
print(f"\nSpearman on held-out 30 elites: {sp_elites:.4f}")


# ============================================================
# 8. Global Spearman on ALL variants
# ============================================================

print("\n=== Global evaluation on all TEM-1 variants ===")

# encode all variants once more (for clarity)
all_variants = df["variant"].tolist()
y_true_all = df["score"].to_numpy()

encoded_all = data_encoder.encode_variants(WT_TEM1, all_variants)
if isinstance(encoded_all, dict):
    encoded_all = encoded_all["encoded_seqs"]
encoded_all = torch.tensor(encoded_all, dtype=torch.long)

ft_model.eval()
all_preds_norm_list = []

with torch.no_grad():
    n = encoded_all.shape[0]
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = encoded_all[start:end].to(DEVICE)
        p = ft_model(batch).cpu().numpy()
        all_preds_norm_list.append(p)

y_pred_all_norm = np.concatenate(all_preds_norm_list)
y_pred_all = y_pred_all_norm * std + mean

assert y_pred_all.shape[0] == y_true_all.shape[0]

sp_full = simple_spearman(y_true_all, y_pred_all)
print(f"Spearman (all TEM-1 variants): {sp_full:.4f}")
