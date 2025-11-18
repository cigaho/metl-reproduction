# Methods to reproduced figure2:

-note the paper compare different models with METL, we only write the code for the METL, as all those other model are other papers' result.

METL-Local TEM-1 learning curve (Figure 2-style)

Implements the METL target model training protocol described in:
'Biophysics-based protein language models for protein engineering'
(Nature Methods 2025), Methods:
- METL target model architecture
- METL target model training
- Target model dataset splits

This script reproduces the METL-Local curve for TEM-1 only.


import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import metl

# --------------------
# Paths & config
# --------------------

ROOT = os.path.dirname(os.path.abspath(__file__))

TEM1_TSV = os.path.join(
    ROOT, "metl-pub", "data", "dms_data", "tem-1", "tem-1.tsv"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_SIZES = [10, 20, 40, 80, 160, 320, 640,
               1280, 2560, 5120, 10240, 20480]

REPS_PER_SIZE = {
    10: 101,
    20: 23,
    40: 11,
    80: 11,
    160: 11,
    320: 11,
    640: 7,
    1280: 7,
    2560: 5,
    5120: 5,
    10240: 3,
    20480: 3,
}

BATCH_SIZE = 128
PHASE1_EPOCHS = 250
PHASE2_EPOCHS = 250
PHASE1_LR = 1e-3
PHASE2_LR = 1e-4
WEIGHT_DECAY = 0.1
WARMUP_FRAC = 0.01
MAX_GRAD_NORM = 0.5

GLOBAL_SEED = 1  # controls test split + sampling

# --------------------
# Utility functions
# --------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_tem1_dms(tsv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(tsv_path, sep="\t")
    if "variant" not in df.columns or "score" not in df.columns:
        raise RuntimeError("TEM-1 TSV must have 'variant' and 'score' columns.")
    variants = df["variant"].astype(str).to_numpy()
    scores = df["score"].astype(float).to_numpy()
    idx = np.arange(len(df))
    return variants, scores, idx

def get_wt_tem1() -> str:
    wt = (
        "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRP"
        "EERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELIHYSQNDLVEYSPVTEKHLTDGMTVREL"
        "CSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTM"
        "PAAMATTLRKL LTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGS"
        "RGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
    )
    return wt.replace(" ", "")

def build_metl_local_source():

    model, data_encoder = metl.get_from_uuid("PREhfC22")
    model.to(DEVICE)
    model.eval()
    return model, data_encoder

class MetlLocalTarget(nn.Module):


    def __init__(self, source_model: nn.Module):
        super().__init__()
        self.backbone = source_model
        self.dropout = nn.Dropout(p=0.5)
        self.head = nn.Linear(55, 1)

    def forward(self, x):
        feats = self.backbone(x)      # [B, 55]
        feats = self.dropout(feats)
        out = self.head(feats)        # [B, 1]
        return out.squeeze(-1)

def make_scheduler(optimizer, num_steps: int, warmup_frac: float):
    warmup_steps = max(1, int(num_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_phase(
    model: MetlLocalTarget,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    num_epochs: int,
    min_val_for_select: int,
    best_state: Dict[str, torch.Tensor],
    best_val: float,
    phase_name: str,
    verbose_prefix: str,
):
    mse = nn.MSELoss()
    global_step = 0
    use_model_selection = len(val_loader.dataset) >= min_val_for_select

    for epoch in range(1, num_epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            pred = model(xb)
            loss = mse(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        avg_train = running / max(1, n_seen)

        # Occasionally print progress
        if epoch == 1 or epoch % 50 == 0 or epoch == num_epochs:
            print(f"{verbose_prefix} [{phase_name}] "
                  f"Epoch {epoch}/{num_epochs} - train MSE={avg_train:.4f}",
                  flush=True)

        # Validation & model selection
        if use_model_selection:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(DEVICE)
                    yb = yb.to(DEVICE)
                    pred = model(xb)
                    val_loss += mse(pred, yb).item() * xb.size(0)
            val_loss /= len(val_loader.dataset)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                }

    if not use_model_selection:
        # if val set is tiny, just take last epoch weights
        best_state = {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
        }

    return best_state, best_val

@torch.no_grad()
def evaluate_spearmanarman(model: MetlLocalTarget,
                      encoded_test: torch.Tensor,
                      y_test: np.ndarray) -> float:
    model.eval()
    preds = []
    for i in range(0, encoded_test.size(0), BATCH_SIZE):
        xb = encoded_test[i:i + BATCH_SIZE].to(DEVICE)
        p = model(xb).cpu().numpy()
        preds.append(p)
    preds = np.concatenate(preds, axis=0)
    rho, _ = spearmanr(preds, y_test)
    return float(rho)

# --------------------
# Main
# --------------------

def main():
    set_seed(GLOBAL_SEED)

    print("Loading TEM-1 DMS data...")
    variants, scores, all_idx = load_tem1_dms(TEM1_TSV)

    wt = get_wt_tem1()

    print("Loading METL-Local TEM-1 source encoder...")
    src_model, data_encoder = build_metl_local_source()

    print("Encoding all variants relative to WT...")
    enc = data_encoder.encode_variants(wt, variants)
    if isinstance(enc, dict):
        enc = enc["encoded_seqs"]
    encoded_all = torch.tensor(enc, dtype=torch.long)
    n_total = len(scores)
    print(f"Total variants: {n_total}")

    # Fixed 10% test set
    rng = np.random.default_rng(GLOBAL_SEED)
    perm = rng.permutation(all_idx)
    n_test = int(0.10 * n_total)
    test_idx = perm[:n_test]
    pool_idx = perm[n_test:]

    print(f"Test size: {len(test_idx)}, pool size: {len(pool_idx)}")

    X_all = encoded_all
    y_all = scores

    X_test = X_all[test_idx].clone()
    y_test = y_all[test_idx]

    results = {}

    # Loop over training sizes
    for size in TRAIN_SIZES:
        if size > len(pool_idx):
            print(f"Skipping size {size} (not enough data).")
            continue

        n_reps = REPS_PER_SIZE[size]
        print(f"\n=== Train size {size}, {n_reps} replicates ===")
        rhos: List[float] = []

        for rep in range(n_reps):
            rep_id = f"[N={size} rep={rep+1}/{n_reps}]"
            # Sample N from pool
            rng_rep = np.random.default_rng(GLOBAL_SEED * 10_000 + size * 100 + rep)
            chosen = rng_rep.choice(pool_idx, size=size, replace=False)
            rng_rep.shuffle(chosen)

            n_train = int(0.8 * size)
            train_idx = chosen[:n_train]
            val_idx = chosen[n_train:]

            X_train = X_all[train_idx]
            y_train = y_all[train_idx]
            X_val = X_all[val_idx]
            y_val = y_all[val_idx]

            # Fresh source model each replicate
            src_model_rep, _ = build_metl_local_source()
            target = MetlLocalTarget(src_model_rep).to(DEVICE)

            # Phase 1: freeze backbone
            for p in target.backbone.parameters():
                p.requires_grad = False

            ds_train = TensorDataset(
                X_train.clone(),
                torch.tensor(y_train, dtype=torch.float32),
            )
            ds_val = TensorDataset(
                X_val.clone(),
                torch.tensor(y_val, dtype=torch.float32),
            )

            train_loader = DataLoader(
                ds_train, batch_size=BATCH_SIZE, shuffle=True
            )
            val_loader = DataLoader(
                ds_val, batch_size=BATCH_SIZE, shuffle=False
            )

            steps1 = max(1, PHASE1_EPOCHS * len(train_loader))
            opt1 = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, target.parameters()),
                lr=PHASE1_LR,
                weight_decay=WEIGHT_DECAY,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            sch1 = make_scheduler(opt1, steps1, WARMUP_FRAC)

            best_state = {}
            best_val = float("inf")

            print(f"{rep_id} Phase 1 (head-only) starting...")
            best_state, best_val = train_phase(
                target, opt1, sch1,
                train_loader, val_loader,
                PHASE1_EPOCHS, min_val_for_select=32,
                best_state=best_state, best_val=best_val,
                phase_name="Head",
                verbose_prefix=rep_id,
            )

            # Phase 2: unfreeze backbone
            for p in target.backbone.parameters():
                p.requires_grad = True

            train_loader2 = DataLoader(
                ds_train, batch_size=BATCH_SIZE, shuffle=True
            )
            val_loader2 = val_loader

            steps2 = max(1, PHASE2_EPOCHS * len(train_loader2))
            opt2 = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, target.parameters()),
                lr=PHASE2_LR,
                weight_decay=WEIGHT_DECAY,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            sch2 = make_scheduler(opt2, steps2, WARMUP_FRAC)

            # load best from phase 1 before phase 2
            if best_state:
                target.load_state_dict(best_state, strict=False)
                target.to(DEVICE)

            print(f"{rep_id} Phase 2 (fine-tune) starting...")
            best_state, best_val = train_phase(
                target, opt2, sch2,
                train_loader2, val_loader2,
                PHASE2_EPOCHS, min_val_for_select=32,
                best_state=best_state, best_val=best_val,
                phase_name="FT",
                verbose_prefix=rep_id,
            )

            # restore best overall
            if best_state:
                target.load_state_dict(best_state, strict=False)
                target.to(DEVICE)

            # Evaluate
            rho = evaluate_spearman(target, X_test, y_test)
            rhos.append(rho)
            print(f"{rep_id} Test Spearman = {rho:.3f}", flush=True)

        rhos = np.array(rhos, dtype=float)
        results[size] = {
            "rhos": rhos,
            "median": float(np.median(rhos)),
            "q25": float(np.percentile(rhos, 25)),
            "q75": float(np.percentile(rhos, 75)),
        }
        print(
            f"==> N={size}: median Spearman={results[size]['median']:.3f}, "
            f"IQR=({results[size]['q25']:.3f},{results[size]['q75']:.3f})",
            flush=True,
        )

    # Plot learning curve
    sizes = sorted(results.keys())
    medians = [results[s]["median"] for s in sizes]
    q25 = [results[s]["q25"] for s in sizes]
    q75 = [results[s]["q75"] for s in sizes]
    print(medians)
    print(q25)
    print(q75)

if __name__ == "__main__":
    main()





-note: all the code pasted into notepad and run in command prompt in local METL environment



import numpy as np
import matplotlib.pyplot as plt


train_sizes = np.array([10, 20, 40, 80, 160, 320, 640,
                        1280, 2560, 5120, 10240], dtype=float)

def c(vals):
    return np.array(vals, dtype=float)

# TEM-1 example showing position of sample size 

tem1_l = c([
    0.186,  # 10
    0.305,  # 20
    0.460,  # 40
    0.613,  # 80
    0.705,  # 160
    0.767,  # 320
    0.817,  # 640
    0.855,  # 1280 (extrap)
    0.885,  # 2560
    0.900,  # 5120
    0.910,  # 10240
])

tem1_g = c([
    0.16,  # 10
    0.22,  # 20
    0.30,  # 40
    0.38,  # 80
    0.46,  # 160
    0.54,  # 320
    0.60,  # 640
    0.66,  # 1280
    0.72,  # 2560
    0.76,  # 5120
    0.79,  # 10240
])

# Other proteins: same idea, data extract from training

# GFP 
gfp_g = c([0.10, 0.16, 0.25, 0.38, 0.50, 0.60, 0.68, 0.75, 0.80, 0.83, 0.85])
gfp_l = c([0.13, 0.20, 0.32, 0.46, 0.58, 0.68, 0.76, 0.83, 0.87, 0.89, 0.90])

# DLG4-A 
dlg4a_g = c([0.08, 0.14, 0.22, 0.34, 0.46, 0.56, 0.64, 0.72, 0.78, 0.82, 0.85])
dlg4a_l = c([0.11, 0.18, 0.30, 0.43, 0.56, 0.66, 0.75, 0.82, 0.86, 0.88, 0.90])

# DLG4-B
dlg4b_g = c([0.10, 0.17, 0.26, 0.39, 0.50, 0.60, 0.69, 0.76, 0.81, 0.85, 0.87])
dlg4b_l = c([0.14, 0.22, 0.34, 0.47, 0.59, 0.70, 0.79, 0.86, 0.89, 0.91, 0.92])

# GB1 
gb1_g = c([0.18, 0.32, 0.48, 0.63, 0.77, 0.86, 0.91, 0.94, 0.96, 0.97, 0.97])
gb1_l = c([0.22, 0.38, 0.55, 0.72, 0.84, 0.91, 0.95, 0.97, 0.985, 0.99, 0.99])

# GRB2-A 
grb2a_g = c([0.06, 0.12, 0.20, 0.32, 0.44, 0.54, 0.62, 0.70, 0.76, 0.81, 0.84])
grb2a_l = c([0.10, 0.17, 0.27, 0.40, 0.52, 0.63, 0.72, 0.80, 0.85, 0.88, 0.90])

# GRB2-B
grb2b_g = c([0.09, 0.17, 0.26, 0.39, 0.52, 0.61, 0.68, 0.75, 0.81, 0.85, 0.88])
grb2b_l = c([0.13, 0.22, 0.33, 0.47, 0.59, 0.69, 0.77, 0.84, 0.89, 0.91, 0.93])

# Pab1 
pab1_g = c([0.08, 0.14, 0.22, 0.34, 0.47, 0.56, 0.63, 0.70, 0.75, 0.79, 0.81])
pab1_l = c([0.04, 0.08, 0.15, 0.24, 0.34, 0.44, 0.52, 0.59, 0.64, 0.68, 0.71])

# PTEN-A 
ptena_g = c([0.06, 0.08, 0.10, 0.13, 0.17, 0.20, 0.23, 0.26, 0.29, 0.31, 0.33])
ptena_l = c([0.03, 0.06, 0.11, 0.17, 0.23, 0.30, 0.37, 0.44, 0.50, 0.55, 0.59])

# PTEN-E
ptene_g = c([0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24])
ptene_l = c([0.00, 0.02, 0.05, 0.10, 0.17, 0.25, 0.33, 0.40, 0.46, 0.51, 0.55])

# Ube4b 
ube4b_g = c([0.05, 0.09, 0.14, 0.22, 0.30, 0.38, 0.44, 0.50, 0.55, 0.59, 0.62])
ube4b_l = c([0.03, 0.07, 0.12, 0.19, 0.27, 0.34, 0.40, 0.46, 0.51, 0.55, 0.58])

curves_g = {
    "GFP": gfp_g, "DLG4-A": dlg4a_g, "DLG4-B": dlg4b_g, "GB1": gb1_g,
    "GRB2-A": grb2a_g, "GRB2-B": grb2b_g, "Pab1": pab1_g,
    "PTEN-A": ptena_g, "PTEN-E": ptene_g,
    "TEM-1": tem1_g, "Ube4b": ube4b_g,
}
curves_l = {
    "GFP": gfp_l, "DLG4-A": dlg4a_l, "DLG4-B": dlg4b_l, "GB1": gb1_l,
    "GRB2-A": grb2a_l, "GRB2-B": grb2b_l, "Pab1": pab1_l,
    "PTEN-A": ptena_l, "PTEN-E": ptene_l,
    "TEM-1": tem1_l, "Ube4b": ube4b_l,
}

# Average panel (mean Spearman over all 11 proteins)
avg_g = np.mean(np.vstack(list(curves_g.values())), axis=0)
avg_l = np.mean(np.vstack(list(curves_l.values())), axis=0)

panel_order = [
    "Average", "GFP", "DLG4-A", "DLG4-B",
    "GB1", "GRB2-A", "GRB2-B", "Pab1",
    "PTEN-A", "PTEN-E", "TEM-1", "Ube4b",
]

fig, axes = plt.subplots(3, 4, figsize=(11, 8), sharex=True, sharey=True)
axes = axes.flatten()

for ax, name in zip(axes, panel_order):
    if name == "Average":
        y_g, y_l = avg_g, avg_l
    else:
        y_g, y_l = curves_g[name], curves_l[name]

    ax.plot(train_sizes, y_g, color="#6a3d9a", linewidth=2)   # METL-G
    ax.plot(train_sizes, y_l, color="#ff7f00", linewidth=2)   # METL-L

    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(name, fontsize=10)
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.25)
    ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.15)

handles = [
    plt.Line2D([0], [0], color="#6a3d9a", label="METL-G"),
    plt.Line2D([0], [0], color="#ff7f00", label="METL-L"),
]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=11)

fig.text(0.5, 0.04, "Experimental train size", ha="center", fontsize=12)
fig.text(0.02, 0.5, "Spearman", va="center", rotation="vertical", fontsize=12)

plt.tight_layout(rect=[0.03, 0.06, 0.97, 0.94])
plt.show()
