# Methods to reproduce figure3a

-this code uses TEM-1 as an exmaple and prints out the 9 spearman and the median spearman which will be used to plot the graph with other protein

import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import metl  # from metl-pretrained package


# ---------- Config ----------
DMS_PATH = "metl-pub/data/dms_data/tem-1/tem-1.tsv"
UUID_LOCAL_TEM1 = "PREhfC22"          # METL-L-2M-1D-TEM-1 local source
N_REPLICATES = 9
MUTATION_TRAIN_FRAC = 0.8
INNER_TRAIN_FRAC = 0.9                # 90% train / 10% val from train-pool
BATCH_SIZE = 256
LR_HEAD = 1e-3
LR_FT = 1e-4
N_EPOCHS_HEAD = 15
N_EPOCHS_FT = 5
PATIENCE = 3                          # early-stop patience for both phases
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- Utilities ----------

def parse_variant(v):
    """Parse a variant like 'M0I,S1G' into list of (wt, pos, mut)."""
    muts = []
    for part in v.split(","):
        part = part.strip()
        wt = part[0]
        mut = part[-1]
        pos = int(part[1:-1])
        muts.append((wt, pos, mut))
    return muts


def spearman_np(y_true, y_pred):
    """Spearman correlation without scipy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    def rank(a):
        # average rank for ties
        tmp = a.argsort()
        ranks = np.empty_like(tmp, dtype=float)
        ranks[tmp] = np.arange(len(a))
        # handle ties
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, ranks)
        avg = sums / counts
        return avg[inv]

    r1 = rank(y_true)
    r2 = rank(y_pred)
    if np.std(r1) == 0 or np.std(r2) == 0:
        return 0.0
    return np.corrcoef(r1, r2)[0, 1]


def build_backbone_and_head():
    """
    Load TEM-1 local source model and wrap it:
    - backbone: embedder + transformer encoder + pooling + fc1
    - head: new Linear(256 -> 1) regression layer
    """
    base_model, data_encoder = metl.get_from_uuid(UUID_LOCAL_TEM1)

    # Extract backbone modules in the order they are applied
    m = base_model.model
    backbone = nn.Sequential(
        m.embedder,
        m.tr_encoder,
        m.avg_pooling,
        m.fc1,
    )

    # Freeze backbone initially
    for p in backbone.parameters():
        p.requires_grad = False

    head = nn.Linear(256, 1)

    model = nn.Sequential(backbone, head)
    return model.to(DEVICE), data_encoder, backbone, head


def encode_variants(data_encoder, wt_seq, variants):
    encoded = data_encoder.encode_variants(wt_seq, variants)
    if isinstance(encoded, dict):
        encoded = encoded["encoded_seqs"]
    return torch.tensor(encoded, dtype=torch.long)


def make_loaders(encoded, scores, train_idx, val_idx):
    X_train = encoded[train_idx]
    y_train = torch.tensor(scores[train_idx], dtype=torch.float32)
    X_val = encoded[val_idx]
    y_val = torch.tensor(scores[val_idx], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


def train_phase(model, loaders, optimizer, max_epochs, phase_name):
    train_loader, val_loader = loaders
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, max_epochs + 1):
        # train
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            preds = model(xb).squeeze(-1)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                preds = model(xb).squeeze(-1)
                loss = loss_fn(preds, yb)
                val_losses.append(loss.item())

        train_mse = float(np.mean(train_losses))
        val_mse = float(np.mean(val_losses))
        print(f"[{phase_name}] Epoch {epoch}/{max_epochs} - "
              f"train MSE: {train_mse:.4f} - val MSE: {val_mse:.4f}")

        if val_mse + 1e-6 < best_val:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


# ---------- Main protocol ----------

def main():
    # 1) Load DMS data
    df = pd.read_csv(DMS_PATH, sep="\t")
    # keep only numeric score
    df = df.dropna(subset=["score"]).reset_index(drop=True)

    variants = df["variant"].tolist()
    scores = df["score"].to_numpy()
    print(f"Total variants in TEM-1 DMS: {len(variants)}")

    # 2) Parse mutations and define global mutation split (train/test)
    all_mut_types = set()
    variant_muts = []  # list of lists of mutation tuples
    for v in variants:
        muts = parse_variant(v)
        variant_muts.append(muts)
        for m in muts:
            all_mut_types.add(m)

    all_mut_types = sorted(all_mut_types)
    n_mut = len(all_mut_types)
    print(f"Total unique mutation types: {n_mut}")
    
    rng = np.random.default_rng(42)  # fixed for reproducibility
    n_train_mut = int(MUTATION_TRAIN_FRAC * n_mut)
    
    # Sample indices, then map back to tuples so they stay hashable
    train_idx = rng.choice(np.arange(n_mut), size=n_train_mut, replace=False)
    train_mut_types = {all_mut_types[i] for i in train_idx}
    test_mut_types = set(all_mut_types) - train_mut_types


    # Assign variants based on mutation types
    train_indices = []
    test_indices = []
    discard_indices = []

    for i, muts in enumerate(variant_muts):
        muts_set = set(muts)
        if muts_set.issubset(train_mut_types):
            train_indices.append(i)
        elif muts_set.issubset(test_mut_types):
            test_indices.append(i)
        else:
            discard_indices.append(i)

    print(f"Train pool: {len(train_indices)} variants")
    print(f"Test pool:  {len(test_indices)} variants")
    print(f"Discarded:  {len(discard_indices)} variants")

    # 3) Encode ALL variants once (we'll index into this)
    # WT TEM-1 sequence (same as before)
    wt_tem1 = (
        "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRP"
        "EERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVREL"
        "CSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTM"
        "PAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGS"
        "RGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
    )

    # Need data_encoder: get once from source model loader
    _, data_encoder, _, _ = build_backbone_and_head()
    encoded_all = encode_variants(data_encoder, wt_tem1, variants)

    # convenient numpy arrays for indexing
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    y_all = scores

    spearmans = []

    # 4) 9 replicates: different 90/10 splits of the TRAIN POOL
    for rep in range(N_REPLICATES):
        print(f"\n=== Replicate {rep+1}/{N_REPLICATES} ===")

        rng_rep = np.random.default_rng(100 + rep)

        # shuffle train pool indices
        perm = rng_rep.permutation(train_indices)
        split = int(INNER_TRAIN_FRAC * len(perm))
        train_sub_idx = perm[:split]
        val_sub_idx = perm[split:]

        # Build model + loaders for this replicate
        model, _, backbone, head = build_backbone_and_head()

        # Phase 1: head-only
        for p in backbone.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True

        train_loader, val_loader = make_loaders(
            encoded_all,
            y_all,
            train_sub_idx,
            val_sub_idx,
        )
        opt_head = torch.optim.Adam(head.parameters(), lr=LR_HEAD)
        best_val_head = train_phase(
            model, (train_loader, val_loader),
            opt_head, N_EPOCHS_HEAD, phase_name="Head"
        )
        print(f"Best val MSE (head-only): {best_val_head:.4f}")

        # Phase 2: unfreeze backbone for gentle fine-tuning
        for p in backbone.parameters():
            p.requires_grad = True
        opt_ft = torch.optim.Adam(model.parameters(), lr=LR_FT)
        best_val_ft = train_phase(
            model, (train_loader, val_loader),
            opt_ft, N_EPOCHS_FT, phase_name="FT"
        )
        print(f"Best val MSE (fine-tune): {best_val_ft:.4f}")

        # Evaluate on full TEST pool
        model.eval()
        with torch.no_grad():
            xb = encoded_all[test_indices].to(DEVICE)
            y_true = y_all[test_indices]
            y_pred = model(xb).squeeze(-1).cpu().numpy()

        sp = spearman_np(y_true, y_pred)
        spearmans.append(sp)
        print(f"Spearman on mutation-extrapolation TEST: {sp:.4f}")

        # cleanup a bit
        del model, backbone, head, opt_head, opt_ft, xb, y_pred

    spearmans = np.array(spearmans)
    print("\nPer-replicate Spearman:", np.round(spearmans, 4).tolist())
    print(f"Median Spearman over {N_REPLICATES} reps: {np.median(spearmans):.4f}")


if __name__ == "__main__":
    main()




import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

data = OrderedDict({
    "GFP":   {"METL-G": [0.61,0.63,0.60,0.62,0.61,0.64,0.60,0.62,0.63],
              "METL-L": [0.62,0.64,0.63,0.62,0.64,0.63,0.65,0.64,0.63]},
    "DLG4-A":{"METL-G": [0.59,0.53,0.57,0.55,0.54,0.56,0.55,0.54,0.57],
              "METL-L": [0.58,0.60,0.62,0.61,0.60,0.61,0.62,0.60,0.59]},
    "DLG4-B":{"METL-G": [0.68,0.70,0.72,0.71,0.70,0.69,0.70,0.71,0.72],
              "METL-L": [0.72,0.73,0.75,0.74,0.74,0.75,0.73,0.74,0.75]},
    "GB1":   {"METL-G": [0.86,0.87,0.90,0.88,0.87,0.88,0.89,0.89,0.87],
              "METL-L": [0.90,0.91,0.92,0.93,0.92,0.92,0.91,0.92,0.92]},
    "GRB2-A":{"METL-G": [0.60,0.61,0.60,0.59,0.60,0.61,0.60,0.61,0.60],
              "METL-L": [0.64,0.65,0.64,0.66,0.65,0.65,0.64,0.66,0.65]},
    "GRB2-B":{"METL-G": [0.70,0.72,0.73,0.73,0.72,0.71,0.72,0.72,0.73],
              "METL-L": [0.76,0.77,0.76,0.75,0.76,0.77,0.76,0.76,0.77]},
    "Pab1":  {"METL-G": [0.56,0.57,0.58,0.58,0.57,0.56,0.57,0.58,0.57],
              "METL-L": [0.69,0.70,0.71,0.72,0.70,0.61,0.62,0.60,0.61]}, 
    "PTEN-A":{"METL-G": [0.62,0.63,0.62,0.61,0.62,0.63,0.62,0.61,0.62],
              "METL-L": [0.66,0.66,0.67,0.66,0.65,0.66,0.66,0.66,0.67]},
    "PTEN-E":{"METL-G": [0.41,0.42,0.43,0.43,0.41,0.42,0.42,0.41,0.42],
              "METL-L": [0.45,0.46,0.47,0.46,0.45,0.46,0.45,0.46,0.47]},
    "TEM-1": {"METL-G": [0.68,0.69,0.70,0.69,0.68,0.69,0.68,0.69,0.70],
              "METL-L": [0.72,0.73,0.74,0.73,0.72,0.73,0.72,0.73,0.74]},
    "Ube4b": {"METL-G": [0.38,0.31,0.39,0.40,0.46,0.45,0.37,0.39,0.48],
              "METL-L": [0.38,0.40,0.49,0.42,0.40,0.29,0.41,0.48,0.41]},
})

def mean_and_err(vals, kind="ci95"):
    x = np.asarray(vals, float)
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    n  = x.size
    if kind == "sd":
        err = sd
    elif kind == "sem":
        err = sd / math.sqrt(n) if n > 0 else 0.0
    elif kind == "ci95":
        # 95% CI using t_{0.975, n-1}
        if n <= 1 or sd == 0.0:
            err = 0.0
        else:
            # t critical for df = n-1 (hardcode small n=9 table to avoid SciPy as version compatiable on my pc is not for other, we need multiple pc to run this)
            T95 = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,
                   10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131}.get(n-1, 1.96)
            err = (sd / math.sqrt(n)) * T95
    else:
        raise ValueError("ERR_KIND must be 'sd', 'sem', or 'ci95'")
    return mu, err

models = ["METL-G", "METL-L"]
colors = {"METL-G": "#6b4cff", "METL-L": "#ff8a2b"}

# Compute per-protein stats
proteins = list(data.keys())
stats = {m: {"mu": [], "err": []} for m in models}
for p in proteins:
    for m in models:
        mu, err = mean_and_err(data[p][m], ERR_KIND)
        stats[m]["mu"].append(mu)
        stats[m]["err"].append(err)

# Build the 'Average' point (mean across proteins of the per-protein means)
avg_mu = {}
avg_err = {}
for m in models:
    mus = np.array(stats[m]["mu"], float)
    # 95% CI across proteins (t with df = P-1)
    mu_bar = float(np.mean(mus))
    sd_bar = float(np.std(mus, ddof=1))
    P = mus.size
    if ERR_KIND == "sd":
        err_bar = sd_bar
    elif ERR_KIND == "sem":
        err_bar = sd_bar / math.sqrt(P)
    else:  # ci95
        T95P = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,
                10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131}.get(P-1, 1.96)
        err_bar = (sd_bar / math.sqrt(P)) if P>1 else 0.0
        err_bar *= T95P if P>1 else 0.0
    avg_mu[m] = mu_bar
    avg_err[m] = err_bar

# Insert Average at the front for plotting
x_labels = ["Average"] + proteins
x = np.arange(len(x_labels))
offset = 0.15

plt.figure(figsize=(11, 4))
for i, m in enumerate(models):
    xs = x + (i - 0.5) * offset
    ys = [avg_mu[m]] + stats[m]["mu"]
    es = [avg_err[m]] + stats[m]["err"]
    plt.errorbar(xs, ys, yerr=es, fmt='o', capsize=3, elinewidth=1.2,
                 color=colors[m], label=m, markersize=6)

plt.xticks(x, x_labels, rotation=20)
plt.ylim(0.0, 1.0)
plt.ylabel("Spearman")
plt.xlabel("Model Name")
plt.title("Fig. 3b - style position extrapolation: METL-Global vs METL-Local")
plt.legend()
plt.tight_layout()
plt.show()
