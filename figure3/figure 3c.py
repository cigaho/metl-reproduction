# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 22:48:58 2025

@author: yokik
"""

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import metl  # from metl-pretrained

# ---------------- Config ----------------
DMS_PATH = "metl-pub/data/dms_data/tem-1/tem-1.tsv"
UUID_LOCAL_TEM1 = "PREhfC22"      # METL-L-2M-1D-TEM-1 local source model
N_REPLICATES = 9
POS_TRAIN_FRAC = 0.8               # 80% of positions for training pool (position-extrapolation)
INNER_TRAIN_FRAC = 0.9             # within that pool: 90% train, 10% val
BATCH_SIZE = 256
LR_HEAD = 1e-3
LR_FT = 1e-4
N_EPOCHS_HEAD = 15
N_EPOCHS_FT = 5
PATIENCE = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------- Utilities ---------------

def parse_variant(v):
    """'M0I,S1G' -> [('M',0,'I'), ('S',1,'G')]"""
    muts = []
    for part in v.split(","):
        part = part.strip()
        wt = part[0]
        mut = part[-1]
        pos = int(part[1:-1])
        muts.append((wt, pos, mut))
    return muts

def spearman_np(y_true, y_pred):
    """Spearman rank corr (no scipy)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    def rank(a):
        order = a.argsort(kind="mergesort")
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(a), dtype=float)
        vals, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, r)
        avg = sums / counts
        return avg[inv]

    r1 = rank(y_true); r2 = rank(y_pred)
    s1 = np.std(r1);    s2 = np.std(r2)
    if s1 == 0 or s2 == 0:
        return 0.0
    return float(np.corrcoef(r1, r2)[0, 1])

def build_backbone_and_head():
    """
    Load TEM-1 local source model and wrap it with a new 256->1 head.
    Backbone = embedder + transformer + pooling + fc1 (256-d).
    """
    base_model, data_encoder = metl.get_from_uuid(UUID_LOCAL_TEM1)
    m = base_model.model
    backbone = nn.Sequential(
        m.embedder,
        m.tr_encoder,
        m.avg_pooling,
        m.fc1,
    )
    for p in backbone.parameters():
        p.requires_grad = False
    head = nn.Linear(256, 1)
    model = nn.Sequential(backbone, head).to(DEVICE)
    return model, data_encoder, backbone, head

def encode_variants(data_encoder, wt_seq, variants):
    enc = data_encoder.encode_variants(wt_seq, variants)
    if isinstance(enc, dict):
        enc = enc["encoded_seqs"]
    return torch.tensor(enc, dtype=torch.long)

def make_loaders(encoded, scores, train_idx, val_idx):
    X_train = encoded[train_idx]
    y_train = torch.tensor(scores[train_idx], dtype=torch.float32)
    X_val = encoded[val_idx]
    y_val = torch.tensor(scores[val_idx], dtype=torch.float32)
    return (
        DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(TensorDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False),
    )

def train_phase(model, loaders, optimizer, max_epochs, phase_name):
    train_loader, val_loader = loaders
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, max_epochs + 1):
        # ---- train ----
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb).squeeze(-1)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ---- val ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb).squeeze(-1)
                loss = loss_fn(preds, yb)
                val_losses.append(loss.item())

        train_mse = float(np.mean(train_losses)) if train_losses else 0.0
        val_mse = float(np.mean(val_losses)) if val_losses else 0.0
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

# --------------- Main -------------------

def main():
    # 1) Load DMS and keep numeric scores
    df = pd.read_csv(DMS_PATH, sep="\t").dropna(subset=["score"]).reset_index(drop=True)
    variants = df["variant"].tolist()
    scores = df["score"].to_numpy()
    print(f"Total variants in TEM-1 DMS: {len(variants)}")

    # 2) POSITION split (position-extrapolation)
    #    - Build the set of all mutated residue indices found in the dataset.
    #    - Randomly assign 80% of positions to the TRAIN pool, 20% to TEST pool.
    #    - Variant goes to TRAIN if all its mutated positions ⊆ train_positions;
    #      to TEST if all positions ⊆ test_positions; otherwise DISCARD.
    all_positions = set()
    variant_positions = []  # list of sets of positions per variant
    for v in variants:
        muts = parse_variant(v)
        pos_set = {pos for (_, pos, _) in muts}
        variant_positions.append(pos_set)
        all_positions.update(pos_set)

    all_positions = sorted(all_positions)
    rng = np.random.default_rng(42)
    n_train_pos = int(POS_TRAIN_FRAC * len(all_positions))
    train_pos = set(all_positions[i] for i in rng.choice(len(all_positions), n_train_pos, replace=False))
    test_pos = set(all_positions) - train_pos

    train_indices, test_indices, discard_indices = [], [], []
    for i, pos_set in enumerate(variant_positions):
        if pos_set.issubset(train_pos):
            train_indices.append(i)
        elif pos_set.issubset(test_pos):
            test_indices.append(i)
        else:
            discard_indices.append(i)

    print(f"Train pool (positions): {len(train_indices)} variants")
    print(f"Test pool  (positions): {len(test_indices)} variants")
    print(f"Discarded (mixed pos):  {len(discard_indices)} variants")

    # WT sequence
    wt_tem1 = (
        "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRP"
        "EERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVREL"
        "CSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTM"
        "PAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGS"
        "RGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
    )

    # 3) Encode all variants once
    _, data_encoder, _, _ = build_backbone_and_head()
    encoded_all = encode_variants(data_encoder, wt_tem1, variants)

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    y_all = scores

    spearmans = []

    # 4) 9 replicates: new 90/10 split within the TRAIN position pool each time
    for rep in range(N_REPLICATES):
        print(f"\n=== Replicate {rep+1}/{N_REPLICATES} ===")
        rng_rep = np.random.default_rng(100 + rep)
        perm = rng_rep.permutation(train_indices)
        split = int(INNER_TRAIN_FRAC * len(perm))
        train_sub_idx = perm[:split]
        val_sub_idx = perm[split:]

        # Build fresh model
        model, _, backbone, head = build_backbone_and_head()

        # Phase 1: head-only
        for p in backbone.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True

        train_loader, val_loader = make_loaders(encoded_all, y_all, train_sub_idx, val_sub_idx)
        opt_head = torch.optim.Adam(head.parameters(), lr=LR_HEAD)
        best_val_head = train_phase(model, (train_loader, val_loader), opt_head, N_EPOCHS_HEAD, phase_name="Head")
        print(f"Best val MSE (head-only): {best_val_head:.4f}")

        # Phase 2: gentle fine-tune (unfreeze backbone)
        for p in backbone.parameters():
            p.requires_grad = True
        opt_ft = torch.optim.Adam(model.parameters(), lr=LR_FT)
        best_val_ft = train_phase(model, (train_loader, val_loader), opt_ft, N_EPOCHS_FT, phase_name="FT")
        print(f"Best val MSE (fine-tune): {best_val_ft:.4f}")

        # Evaluate on the position-extrapolation TEST pool
        model.eval()
        with torch.no_grad():
            xb = encoded_all[test_indices].to(DEVICE)
            y_true = y_all[test_indices]
            y_pred = model(xb).squeeze(-1).cpu().numpy()

        sp = spearman_np(y_true, y_pred)
        spearmans.append(sp)
        print(f"Spearman on position-extrapolation TEST: {sp:.4f}")

        del model, backbone, head, opt_head, opt_ft

    spearmans = np.array(spearmans)
    print("\nPer-replicate Spearman:", np.round(spearmans, 4).tolist())
    print(f"Median Spearman over {N_REPLICATES} reps: {np.median(spearmans):.4f}")

if __name__ == "__main__":
    main()

