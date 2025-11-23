# About our extension

In protein engineering, the most important things that we care about is whether we can find the best mutations base on our fitness definition (for example, in case of TEM-1 it is the ability to degrade beta-lactam antibiotics). In the paper, they did something that tries to address this which is figure 3d, score extrapolation and showed that if our model never met the best mutation (during fine tuning with experimental data set), it will not be able to recognize one. We want to see if this can be addressed. 

We first want to use a better measure, instead of the spearman, we choose to use the recall, so how many from the top mutation, does our model remember. We remove the top 20 mutation from the experiment data set (can be changed) and use that chopped experimental to fine tune the the pretrained model. Then ask the model to predict how good were those top20 mutations. And as expected they did bad, but we made some mistake during this as well, for example, doing the train test split on the sorted experimental data. We try something simple, by giving more weight to the mutation sequence with score>0 so model pays more attentions to them and we also use a non linear head to capture more complexity, and the recall improve, although not big, but we showed that it is possible to let the model 'understand' what is a good mutation before see it. We initially have a lot of idea to this, mostly have something to do with the pertaining, for example not all 55 biophysics attribute maybe contributing to fitness, some of them might just be noise, and we can also let the model to pay more attention to certain positions, for example in our TEM-1 example, certain amino acid position may contribute more to the catalytic function like ones in active site or ones that is for substrate binding and stabilization. This is specific to different enzymes. But unfortunately those cannot be done due to pyrosseta not being available, we need to apply for a license that require 1 month for them to approve.

Biophysics prior for protein engineering is a very good addition to the esting evolutionary prior (like linear EVE), we have also shows a very interesting result in addition on the detail proof the authors gives: by running the TEM-1 fine tuned model that is fine tuned on double mutation data set only, on a independent dataset Firnberg et al., 2014, which contain single mutations only. This is shown during presentation, and we can see although that the prediction score is very bad, due to different fitness definition as Firnberg et al., 2014 test those mutation on several different ampicillin concentration but the one that our paper use is only for 1 ampicillin concentration. But the model can distinguish in general between the different of a good mutation and a bad mutation, as you can see the trend of the position gradient line, but if we run the recall on this, it is very bad, but the spearman sees very good- 0.8524, due to the model not be able to distinguish how good a mutations is and how bad a mutations is. 

The code for the elite held out will be given seperately, here is the code for the firnberg dataset, including the data processing detail we did not have time to talk about.

import os
import torch
import pandas as pd
import metl
from scipy.stats import pearsonr, spearmanr

# ---------- 1. Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "external_data",
    "firnberg_tem1",
    "Data_S1-S4.xlsx"
)

# ---------- 2. Load Firnberg S2 sheet ----------
# This sheet contains per-mutation fitnesses.
sheet_name = "S2 Missense mutation fitnesses"

df_raw = pd.read_excel(DATA_PATH, sheet_name=sheet_name)
print("Raw Firnberg rows:", len(df_raw))
print("Columns:", list(df_raw.columns))

# We now auto-detect important columns by name fragments
def find_col(possible_names):
    cols = list(df_raw.columns)
    for cand in possible_names:
        for c in cols:
            if cand.lower() in str(c).lower():
                return c
    raise ValueError(f"Could not find any of: {possible_names}")

COL_AMBLER = find_col(["ambler"])
COL_WT     = find_col(["wt aa", "wild-type aa", "wt_aa"])
COL_MUT    = find_col(["mutant aa", "mut aa", "mut_aa"])
COL_FIT    = find_col(["fitness"])

print("Using columns:")
print("  Ambler position :", COL_AMBLER)
print("  WT AA           :", COL_WT)
print("  Mutant AA       :", COL_MUT)
print("  Fitness         :", COL_FIT)

df = df_raw[[COL_AMBLER, COL_WT, COL_MUT, COL_FIT]].copy()
df = df.dropna(subset=[COL_AMBLER, COL_WT, COL_MUT, COL_FIT])

# Keep only proper single AA substitutions (no stops, no weird)
df[COL_WT]  = df[COL_WT].astype(str).str.strip()
df[COL_MUT] = df[COL_MUT].astype(str).str.strip()

df = df[(df[COL_WT].str.len() == 1) &
        (df[COL_MUT].str.len() == 1) &
        (df[COL_MUT] != "*")]

df[COL_AMBLER] = df[COL_AMBLER].astype(int)

print("After filtering to missense single AAs:", len(df))

# ---------- 3. WT TEM-1 sequence (full, including signal peptide) ----------
wt_tem1 = (
    "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRP"
    "EERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVREL"
    "CSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTM"
    "PAAMATTLRKL LTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGS"
    "RGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
).replace(" ", "")

print("WT TEM-1 length:", len(wt_tem1))

# ---------- 4. Infer mapping: Ambler position -> index in wt_tem1 ----------
# For each row, find an index i where wt_tem1[i] == WT AA.
# Compute offset = i - Ambler. The correct mapping should have one dominant offset.

offset_counts = {}

for _, row in df.iterrows():
    amb = int(row[COL_AMBLER])
    wt = row[COL_WT]
    # all indices where wt_tem1 matches this wt AA
    idxs = [i for i, aa in enumerate(wt_tem1) if aa == wt]
    # compute offsets for this row
    for i in idxs:
        off = i - amb
        offset_counts[off] = offset_counts.get(off, 0) + 1

# pick the most common offset
if not offset_counts:
    raise ValueError("Could not compute any offsets; check WT AA column.")

best_offset = max(offset_counts, key=offset_counts.get)
print("Inferred Ambler->index offset:", best_offset,
      "| count:", offset_counts[best_offset])

# quick sanity: require that this offset is strongly dominant
total_votes = sum(offset_counts.values())
if offset_counts[best_offset] < 0.5 * total_votes:
    print("WARNING: Ambler offset not clearly dominant. Check alignment carefully.")

# ---------- 5. Build METL-style variant strings ----------
def make_variant(row):
    amb = int(row[COL_AMBLER])
    wt = row[COL_WT]
    mut = row[COL_MUT]
    idx = amb + best_offset   # this should be 0-based index into wt_tem1
    # sanity: check wt_tem1 at idx matches expected wt
    if idx < 0 or idx >= len(wt_tem1) or wt_tem1[idx] != wt:
        # if mismatch, return None and we'll drop later
        return None
    return f"{wt}{idx}{mut}"

df["variant"] = df.apply(make_variant, axis=1)
before = len(df)
df = df[df["variant"].notnull()].copy()
after = len(df)
print(f"Kept {after} / {before} rows after enforcing WT-AA consistency")

# ---------- 6. Load METL-Local TEM-1 model ----------
model, data_encoder = metl.get_from_uuid("64ncFxBR")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# ---------- 7. Encode variants ----------
variants = df["variant"].tolist()
encoded = data_encoder.encode_variants(wt_tem1, variants)

if isinstance(encoded, dict):
    encoded_seqs = encoded["encoded_seqs"]
else:
    encoded_seqs = encoded

encoded_tensor = torch.tensor(encoded_seqs, dtype=torch.long)

# ---------- 8. Run model in batches ----------
batch_size = 512
all_preds = []

with torch.no_grad():
    for start in range(0, encoded_tensor.shape[0], batch_size):
        end = min(start + batch_size, encoded_tensor.shape[0])
        batch = encoded_tensor[start:end].to(device)
        batch_preds = model(batch).squeeze().view(-1).cpu()
        all_preds.append(batch_preds)

all_preds = torch.cat(all_preds).numpy()
print("Preds length:", len(all_preds))

df["metl_pred"] = all_preds

# ---------- 9. Compute correlation ----------
fit = df[COL_FIT].astype(float)

pearson = pearsonr(df["metl_pred"], fit)[0]
spearman = spearmanr(df["metl_pred"], fit)[0]

print(f"Firnberg external test Pearson r:  {pearson:.4f}")
print(f"Firnberg external test Spearman r: {spearman:.4f}")

# ---------- 10. Save merged table ----------
out_path = os.path.join(BASE_DIR, "firnberg_tem1_metl_predictions.csv")
df.to_csv(out_path, index=False)
print("Saved merged data to", out_path)
print(df.head())

import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))
plt.scatter(df["metl_pred"], df[COL_FIT], s=5, alpha=0.3)
plt.xlabel("METL-Local predicted fitness")
plt.ylabel("Firnberg experimental fitness")
plt.title("TEM-1 single mutants (Firnberg 2014): METL-Local vs experiment")
plt.tight_layout()
plt.savefig("firnberg_tem1_metl_scatter.png", dpi=300)
print("Saved figure: firnberg_tem1_metl_scatter.png")





