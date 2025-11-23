# Figure 2 — METL performance over experiment data size**

**Goal:** evaluate performance of METL against baseline methods in data-scarce protein engineering scenarios. Validating feasibility for adopting biophysics as prior in protein engineering

**How we did it:** we systematically varied training set sizes (from 8 to 16,384 samples) across 11 protein fitness datasets and measured test set performance using Spearman correlation. Methods included biophysics-based (METL-Global, METL-Local), evolutionary (ESM-2, EVE), and simple baselines (Linear, Rosetta). In our reproduction, we focus on the biophysics-based METL, which is purposed by the paper. Learning curves on log-log scales reveal how each method's inductive biases affect sample efficiency, with METL-Local excelling in extreme low-data regimes due to its protein-specific biophysical pretraining.

---

**Key Findings:**
- **Protein-specific patterns:** different proteins show varying optimal method preferences, highlighting the context-dependent value of different prior knowledge types
- **Sample efficiency:** METL frameworks maintain competitive performance within scarce dataset than general-purpose models when it comes to protein that is strongly related to biophysics features

The reproduction confirms the original paper's conclusion that biophysical priors provide critical advantages for practical protein engineering where experimental data is severely limited.




-note the paper compare different models with METL, we only write the code for the METL, as all those other model are other papers' result.

