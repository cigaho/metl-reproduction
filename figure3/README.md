#Figure 3a — Mutation-type extrapolation

Goal: can the model predict fitness for mutation types it has never seen?
How we did it: we parse each variant into mutation tuples (wt_aa, position, mut_aa). We then split the set of mutation types into disjoint train/test pools (e.g., 80%/20%). A variant goes to the train pool only if all of its mutation types are in the train set; it goes to test only if all are in the test set; “mixed” variants are discarded. We encode sequences once with the METL-Local TEM-1 encoder, then train a small regression head on top of the frozen backbone (phase-1), followed by a light fine-tune (phase-2). For each replicate we do a 90/10 train/val split within the train pool, pick the best epoch by val MSE, and report Spearman ρ on the mutation-type-disjoint test set; the figure plots the median (and IQR) over 9 replicates.

#Figure 3b — Position (site) extrapolation

Goal: can the model generalize to unseen sequence positions?
How we did it: from the double-mutant file we extract the unordered site indices (i, j) for each variant. We randomly split the set of positions into train vs test (e.g., 80%/20%). A variant belongs to train only if both sites lie in the train-site set; it belongs to test only if both sites lie in the test-site set; “cross” pairs are discarded. Training/inference are identical to 3a (encode once; head-only phase then a short fine-tune; 90/10 inner split; best-val checkpoint). We evaluate Spearman ρ on variants whose sites are entirely unseen during training and store the 9-replicate median.

#Figure 3c — Combination (pair) extrapolation

Goal: can the model compose known parts: i.e., predict doubles where each single-site mutation type has been seen before, but their combination has never been seen?
How we did it: we keep doubles only. We ensure each individual mutation type (e.g., A42G and L85F) appears in the training set somewhere (possibly paired with other sites), but the specific unordered pair of mutation types (or, equivalently, the exact unordered site-pair with those identities) is held out for test. Concretely: build the set of observed double combinations from the data, sample a train subset of combinations, then define the test set as doubles whose exact combination is unseen while their component mutation types are present elsewhere in train. Train as in 3a (encode once, two-phase training, inner 90/10 split) and report Spearman ρ on the unseen-combination test set, aggregating the 9-replicate median.

#Figure 3d — Order extrapolation (singles → higher order)

Goal: can the model trained only on single mutants predict multi-mutant fitness?
How we did it in principle: take all single mutants as the training set; hold out all ≥2-mutant variants for testing; train the same two-phase target on singles and compute Spearman ρ on higher-order mutants.
TEM-1 wrinkle: the provided TEM-1 TSV contains only doubles. So true order-extrapolation can’t be run on this file. For TEM-1 we instead used a double-only proxy that still measures combinatorial generalization: train on a subset of unordered site-pairs and test on disjoint site-pairs (unseen pairs). The training procedure, validation, and reporting (median over 9 replicates) stay identical to the other panels.
