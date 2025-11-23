
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import metl

# ---------------- Config ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DMS_PATH = os.path.join(ROOT, "metl-pub", "data", "dms_data", "tem-1", "tem-1.tsv")
UUID_LOCAL_TEM1 = "PREhfC22"   # METL-Local TEM-1 source
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE      = 256
LR_HEAD         = 1e-3
LR_FT           = 1e-4
EPOCHS_HEAD     = 15
EPOCHS_FT       = 5
REPLICATES      = 9
PAIR_TRAIN_FRAC = 0.8    # 80% of unique unordered site-pairs form the train pool
INNER_TRAIN_FRAC = 0.9   # 90% train / 10% val inside the train pool

# -------------- Small utils --------------
def wt_tem1():
    wt = (
        "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRP"
        "EERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVREL"
        "CSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTM"
        "PAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGS"
        "RGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
    )
    return wt.replace(" ", "")

def parse_variant_positions(v):
    # 'M0I,S1G' -> unordered site-pair (i,j)
    pos = []
    for part in v.split(","):
        part = part.strip()
        pos.append(int(part[1:-1]))
    if len(pos) != 2:
        raise ValueError(f"Expected a double mutant, got: {v}")
    i, j = sorted(pos)
    return (i, j)

def encode_variants(encoder, wt_seq, variants):
    enc = encoder.encode_variants(wt_seq, variants)
    if isinstance(enc, dict):
        enc = enc["encoded_seqs"]
    return torch.tensor(enc, dtype=torch.long)

def spearman_np(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    def rank(a):
        order = a.argsort()
        r = np.empty_like(order, dtype=float); r[order] = np.arange(len(a))
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, r); avg = sums / cnt
        return avg[inv]
    r1, r2 = rank(y_true), rank(y_pred)
    if r1.std() == 0 or r2.std() == 0: return 0.0
    return float(np.corrcoef(r1, r2)[0, 1])

def build_backbone_and_head():
    # METL-Local TEM-1 source -> [B, 55]; add Linear(55->1) head
    base_model, data_encoder = metl.get_from_uuid(UUID_LOCAL_TEM1)
    backbone = base_model
    for p in backbone.parameters():
        p.requires_grad = False
    head = nn.Linear(55, 1)
    model = nn.Sequential(backbone, head).to(DEVICE)
    return model, data_encoder, backbone, head

def make_loaders(X, y, idx_train, idx_val):
    Xtr = X[idx_train]; ytr = torch.tensor(y[idx_train], dtype=torch.float32)
    Xva = X[idx_val];   yva = torch.tensor(y[idx_val], dtype=torch.float32)
    tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    va = DataLoader(TensorDataset(Xva, yva), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    return tr, va

def train_with_progress(model, loaders, optimizer, epochs, phase_name):
    """Print per-epoch progress exactly like your 3a format."""
    train_loader, val_loader = loaders
    crit = nn.MSELoss()
    best = None; best_val = float("inf")
    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = crit(pred, yb)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())
        avg_train = float(np.mean(tr_losses)) if tr_losses else float("inf")

        # --- validate ---
        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                pred = model(xb).squeeze(-1)
                va_losses.append(crit(pred, yb).item())
        avg_val = float(np.mean(va_losses)) if va_losses else float("inf")

        print(f"[{phase_name}] Epoch {epoch}/{epochs} - "
              f"train MSE: {avg_train:.4f} - val MSE: {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            best = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best is not None:
        model.load_state_dict(best)
    return best_val

# -------------- Main --------------
def main():
    # 1) Load double-only DMS
    df = pd.read_csv(DMS_PATH, sep="\t").dropna(subset=["score"]).reset_index(drop=True)
    if not (df["num_mutations"] == 2).all():
        raise RuntimeError("This script assumes the TEM-1 TSV contains ONLY double mutants.")

    # 2) Site-pair extraction
    pairs = df["variant"].map(parse_variant_positions).to_numpy()
    # robust uniq for tuples:
    uniq = np.unique(np.array(pairs.tolist(), dtype=[("i", int), ("j", int)]))
    unique_pairs = np.array([(p["i"], p["j"]) for p in uniq])
    print(f"Total variants in TEM-1 DMS: {len(df)}")
    print(f"Unique unordered site-pairs: {len(unique_pairs)}")

    # 3) Encode all once
    wt = wt_tem1()
    _, encoder, _, _ = build_backbone_and_head()
    encoded_all = encode_variants(encoder, wt, df["variant"].tolist())
    y_all = df["score"].to_numpy()

    spearmans = []

    # 4) Replicates
    for rep in range(REPLICATES):
        print(f"\n=== Replicate {rep+1}/{REPLICATES} ===")
        rng = np.random.default_rng(1000 + rep)

        perm = rng.permutation(len(unique_pairs))
        n_train_pairs = int(PAIR_TRAIN_FRAC * len(unique_pairs))
        train_pair_set = set(map(tuple, unique_pairs[perm[:n_train_pairs]]))
        test_pair_set  = set(map(tuple, unique_pairs[perm[n_train_pairs:]]))

        idx_train_pool = np.array([i for i, p in enumerate(pairs) if p in train_pair_set])
        idx_test       = np.array([i for i, p in enumerate(pairs) if p in test_pair_set])

        perm_tr = rng.permutation(idx_train_pool)
        split = int(INNER_TRAIN_FRAC * len(perm_tr))
        idx_train = perm_tr[:split]
        idx_val   = perm_tr[split:]

        print(f"Train pool (pairs): {len(train_pair_set)} | Test pairs: {len(test_pair_set)}")
        print(f"Train variants: {len(idx_train)} | Val: {len(idx_val)} | Test (unseen pairs): {len(idx_test)}")

        # 5) Model
        model, _, backbone, head = build_backbone_and_head()

        # Phase 1: head-only
        for p in backbone.parameters(): p.requires_grad = False
        for p in head.parameters():     p.requires_grad = True
        train_loader, val_loader = make_loaders(encoded_all, y_all, idx_train, idx_val)
        opt1 = torch.optim.Adam(head.parameters(), lr=LR_HEAD)
        best_val_head = train_with_progress(model, (train_loader, val_loader), opt1, EPOCHS_HEAD, phase_name="Head")
        print(f"Best val MSE (head-only): {best_val_head:.4f}")

        # Phase 2: light fine-tune
        for p in backbone.parameters(): p.requires_grad = True
        opt2 = torch.optim.Adam(model.parameters(), lr=LR_FT)
        best_val_ft = train_with_progress(model, (train_loader, val_loader), opt2, EPOCHS_FT, phase_name="FT")
        print(f"Best val MSE (fine-tune): {best_val_ft:.4f}")

        # 6) Evaluate on unseen-pair test set
        model.eval()
        with torch.no_grad():
            xb = encoded_all[idx_test].to(DEVICE)
            y_true = y_all[idx_test]
            y_pred = model(xb).squeeze(-1).cpu().numpy()
        sp = spearman_np(y_true, y_pred)
        spearmans.append(sp)
        print(f"Spearman on unseen site-pair TEST: {sp:.4f}")

        del model, backbone, head, train_loader, val_loader, opt1, opt2

    spearmans = np.array(spearmans, float)
    print("\nPer-replicate Spearman:", np.round(spearmans, 4).tolist())
    print(f"Median Spearman over {REPLICATES} reps: {np.median(spearmans):.4f}")

if __name__ == "__main__":
    main()

