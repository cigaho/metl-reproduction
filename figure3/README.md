# Methods to reproduce figure3

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
