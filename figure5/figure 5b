

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import metl

# -------------------- Paths / UUIDs --------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
GB1_TSV = os.path.join(ROOT, "metl-pub", "data", "dms_data", "gb1", "gb1.tsv")
GB1_PDB = os.path.join(ROOT, "metl-pub", "data", "pdb_files", "2GB1.pdb")

# Source checkpoints (replace BIND uuid if you have it)
UUID_METL_L_GB1    = "epegcFiH"     # METL-L-2M-3D-GB1 (Rosetta-only pretrain)
UUID_METL_BIND_GB1 = "PUT_YOURS"    # METL-Bind (GB1-IgG); leave as-is to skip

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_SIZES = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480]
REPS_PER_SIZE = {
    10:101, 20:23, 40:11, 80:11, 160:11, 320:11, 640:7,
    1280:7, 2560:5, 5120:5, 10240:3, 20480:3
}

BATCH_SIZE = 128
PHASE1_EPOCHS = 250
PHASE2_EPOCHS = 250
PHASE1_LR = 1e-3
PHASE2_LR = 1e-4
WEIGHT_DECAY = 0.1
WARMUP_FRAC = 0.01
MAX_GRAD_NORM = 0.5
GLOBAL_SEED = 1

# --- WT sequence (56 aa). We also normalize to strip hidden chars. ---
_GB1_WT_RAW = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDAATKTFTVTE"
def _clean_seq(s: str) -> str:
    return s.replace(" ", "").replace("\n", "").replace("\r", "").strip()

GB1_WT = _clean_seq(_GB1_WT_RAW)   # must be 56

# -------------------- Utilities --------------------
def set_seed(s: int):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_gb1(tsv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(tsv_path, sep="\t")
    if "variant" not in df.columns or "score" not in df.columns:
        raise RuntimeError("gb1.tsv must have columns: variant, score")
    return df["variant"].astype(str).to_numpy(), df["score"].astype(float).to_numpy(), np.arange(len(df))

def validate_variants(variants: np.ndarray, wt_len: int):
    bad = []
    for v in variants:
        # variants are like "E27K,V39A" (1-based positions)
        for part in v.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                pos = int(part[1:-1])  # 1-based
            except Exception:
                bad.append((v, "parse"))
                continue
            if pos < 1 or pos > wt_len:
                bad.append((v, f"pos={pos} out of [1,{wt_len}]"))
    if bad:
        ex = "; ".join([f"{v}:{why}" for v, why in bad[:5]])
        raise ValueError(f"{len(bad)} variants have invalid positions. e.g., {ex}")

class MetlTarget(nn.Module):
    """Backbone = METL source model (55-d feats); Head = Dropout(0.5)+Linear(55->1)."""
    def __init__(self, source_model: nn.Module, pdb_path: str):
        super().__init__()
        self.backbone = source_model
        self.dropout  = nn.Dropout(p=0.5)
        self.head     = nn.Linear(55, 1)
        self.pdb_path = pdb_path

    def forward(self, x):
        feats = self.backbone(x, pdb_fn=self.pdb_path)  # IMPORTANT: pass PDB to relative attention
        feats = self.dropout(feats)
        return self.head(feats).squeeze(-1)

def make_scheduler(optimizer, total_steps: int):
    warm = max(1, int(total_steps * WARMUP_FRAC))
    def lr_lambda(step):
        if step < warm:  # linear warmup
            return float(step + 1) / float(warm)
        if step >= total_steps:
            return 0.0
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1.0 + np.cos(np.pi * prog))  # cosine decay
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_phase(model: MetlTarget, optimizer, scheduler,
                train_loader, val_loader, num_epochs: int,
                min_val_for_select: int, best_state: Dict[str, torch.Tensor],
                best_val: float, phase_name: str, tag: str):
    mse = nn.MSELoss()
    use_select = len(val_loader.dataset) >= min_val_for_select
    for epoch in range(1, num_epochs+1):
        model.train()
        run, n = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            pred = model(xb)
            loss = mse(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            if scheduler is not None: scheduler.step()
            run += loss.item() * xb.size(0); n += xb.size(0)
        avg_train = run / max(1, n)

        # progress line—like your Fig.3 logs
        if epoch == 1 or epoch % 50 == 0 or epoch == num_epochs:
            if use_select:
                model.eval()
                val = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                        val += mse(model(xb), yb).item() * xb.size(0)
                val /= len(val_loader.dataset)
                print(f"{tag} [{phase_name}] Epoch {epoch}/{num_epochs} - train MSE={avg_train:.4f} - val MSE={val:.4f}")
                if val < best_val:
                    best_val = val
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            else:
                print(f"{tag} [{phase_name}] Epoch {epoch}/{num_epochs} - train MSE={avg_train:.4f}")
    if not use_select:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return best_state, best_val

@torch.no_grad()
def eval_spearman(model: MetlTarget, X_test: torch.Tensor, y_test: np.ndarray) -> float:
    model.eval()
    preds = []
    for i in range(0, X_test.size(0), BATCH_SIZE):
        xb = X_test[i:i+BATCH_SIZE].to(DEVICE)
        preds.append(model(xb).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    rho, _ = spearmanr(preds, y_test)
    return float(rho)

def encode_all(encoder, wt: str, variants: np.ndarray) -> torch.Tensor:
    enc = encoder.encode_variants(wt, list(variants))
    if isinstance(enc, dict): enc = enc["encoded_seqs"]
    return torch.tensor(enc, dtype=torch.long)

def split_fixed_test(n_total: int, idx: np.ndarray, frac: float = 0.10, seed: int = 1):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(idx)
    n_test = int(frac * n_total)
    return perm[:n_test], perm[n_test:]

def make_loaders(X_all, y_all, train_idx, val_idx):
    Xtr, ytr = X_all[train_idx], y_all[train_idx]
    Xva, yva = X_all[val_idx], y_all[val_idx]
    ds_tr = TensorDataset(Xtr.clone(), torch.tensor(ytr, dtype=torch.float32))
    ds_va = TensorDataset(Xva.clone(), torch.tensor(yva, dtype=torch.float32))
    return (
        DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=False),
        DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, drop_last=False),
    )

def run_one_source(source_uuid: str, label: str) -> Dict[int, Dict[str, np.ndarray]]:
    # Load source + encoder once for encoding (fresh backbone per replicate later)
    src0, encoder = metl.get_from_uuid(source_uuid)
    src0.to(DEVICE); src0.eval()

    # Load + validate DMS
    variants, scores, idx = load_gb1(GB1_TSV)
    print(f"\n=== {label} | total={len(variants):,} ===")

    # Safety: ensure WT length and all variant positions in range
    wt_len = len(GB1_WT)
    print(f"WT length after cleaning = {wt_len}")
    if wt_len != 56:
        raise AssertionError(f"GB1 WT must be 56 aa, got {wt_len}. "
                             f"String (debug): '{GB1_WT}'")
    validate_variants(variants, wt_len)

    print("Encoding GB1 variants relative to WT …")
    X_all = encode_all(encoder, GB1_WT, variants)
    y_all = scores
    n_total = len(y_all)

    # Fixed 10% test split
    test_idx, pool_idx = split_fixed_test(n_total, idx, frac=0.10, seed=GLOBAL_SEED)
    print(f"Fixed test={len(test_idx)}, pool={len(pool_idx)}")
    X_test = X_all[test_idx].clone()
    y_test = y_all[test_idx]

    results = {}
    for N in TRAIN_SIZES:
        if N > len(pool_idx):
            print(f"[{label}] N={N} skipped (pool too small)")
            continue
        reps = REPS_PER_SIZE[N]
        print(f"\n[{label}] N={N} | {reps} replicates")

        rhos: List[float] = []
        for rep in range(reps):
            tag = f"[{label} N={N} rep={rep+1}/{reps}]"
            rng = np.random.default_rng(GLOBAL_SEED*10_000 + N*100 + rep)
            chosen = rng.choice(pool_idx, size=N, replace=False)
            rng.shuffle(chosen)
            ntr = int(0.8 * N)
            tr_idx, va_idx = chosen[:ntr], chosen[ntr:]

            # Fresh backbone per replicate
            src, _ = metl.get_from_uuid(source_uuid)
            target = MetlTarget(src, GB1_PDB).to(DEVICE)

            # Phase 1 (head only)
            for p in target.backbone.parameters(): p.requires_grad = False
            tr_loader, va_loader = make_loaders(X_all, y_all, tr_idx, va_idx)
            steps1 = max(1, PHASE1_EPOCHS * len(tr_loader))
            opt1 = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, target.parameters()),
                lr=PHASE1_LR, weight_decay=WEIGHT_DECAY, betas=(0.9,0.999), eps=1e-8
            )
            sch1 = make_scheduler(opt1, steps1)
            best_state, best_val = {}, float("inf")
            print(f"{tag} Phase 1 (head-only) starting...")
            best_state, best_val = train_phase(
                target, opt1, sch1, tr_loader, va_loader,
                PHASE1_EPOCHS, min_val_for_select=32,
                best_state=best_state, best_val=best_val,
                phase_name="Head", tag=tag
            )

            # Phase 2 (fine-tune)
            for p in target.backbone.parameters(): p.requires_grad = True
            steps2 = max(1, PHASE2_EPOCHS * len(tr_loader))
            opt2 = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, target.parameters()),
                lr=PHASE2_LR, weight_decay=WEIGHT_DECAY, betas=(0.9,0.999), eps=1e-8
            )
            sch2 = make_scheduler(opt2, steps2)
            if best_state:
                target.load_state_dict(best_state, strict=False); target.to(DEVICE)
            print(f"{tag} Phase 2 (fine-tune) starting...")
            best_state, best_val = train_phase(
                target, opt2, sch2, tr_loader, va_loader,
                PHASE2_EPOCHS, min_val_for_select=32,
                best_state=best_state, best_val=best_val,
                phase_name="FT", tag=tag
            )
            if best_state:
                target.load_state_dict(best_state, strict=False); target.to(DEVICE)

            # Evaluate on the fixed test set
            rho = eval_spearman(target, X_test, y_test)
            rhos.append(rho)
            print(f"{tag} Test Spearman = {rho:.3f}")

            del target, src, opt1, opt2

        rhos = np.asarray(rhos, float)
        results[N] = dict(
            rhos=rhos,
            median=float(np.median(rhos)),
            q25=float(np.percentile(rhos,25)),
            q75=float(np.percentile(rhos,75)),
        )
        print(f"==> [{label}] N={N}: median={results[N]['median']:.3f}  "
              f"IQR=({results[N]['q25']:.3f},{results[N]['q75']:.3f})")
    return results

def main():

    set_seed(GLOBAL_SEED)

    # Log WT length to catch hidden whitespace issues immediately
    print(f"Cleaned GB1 WT length = {len(GB1_WT)}")
    if len(GB1_WT) != 56:
        print(f"[DEBUG] WT string is: '{GB1_WT}'")

    # Run METL-L(GB1)
    res_l = run_one_source(UUID_METL_L_GB1, label="METL-L(GB1)")

    # Save raw results
    out_npz = os.path.join(ROOT, "fig5b_results.npz")
    np.savez_compressed(out_npz, metl_l=res_l)
    print(f"\nSaved results to {out_npz}")

if __name__ == "__main__":
    main()
