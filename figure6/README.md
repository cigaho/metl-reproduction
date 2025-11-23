# Figure 6 - Validation of the METL pipeline

# Explain figure 6

This figure (Figure 6) presents a critical real-world validation of the METL pipeline by transitioning from in silico prediction to the actual design and experimental characterization of novel green fluorescent protein (GFP) variants. The central question it addresses is whether a model fine-tuned on an extremely limited experimental dataset (N=64) can successfully guide the design of functional proteins, a scenario that mirrors the practical constraints of protein engineering. The validation is structured as a controlled comparison. The left panel outlines the complete design workflow, which begins with pretraining a model on a vast corpus of simulated GFP variants to instill a biophysical prior. This model is then fine-tuned on the small set of real experimental measurements. Finally, the optimized model is used to design new protein sequences. The right panel directly contrasts the experimental outcomes of two distinct design strategies. The top chart (c) displays the measured relative brightness of sequences designed by METL. A significant number of these variants, particularly those based on "observed" mutation types, exhibit high fluorescence, demonstrating that the model successfully learned to extrapolate from minimal data to propose viable designs. In stark contrast, the bottom chart (d) shows the results from a set of randomly designed variants, which serve as a crucial negative control. The overwhelming majority of these random sequences show little to no fluorescence. This side-by-side comparison provides compelling evidence that the functional success of the METL-designed sequences is not a product of chance but is a direct result of the model's learned sequence-function relationships, thereby validating the practical utility of the biophysics-based prior in a genuine low-data design challenge.

# Figure reproduction

**Figure 6c — METL-designed variants brightness**

**Goal:** evaluate the functional success of protein sequences designed by METL under extreme low-data conditions (N=64 experimental training examples).  
**How we did it:** we fine-tuned METL (pretrained on 20M simulated examples) with only 64 experimental measurements, then used simulated annealing to design novel variants requiring mutation extrapolation (unobserved amino acids) and regime extrapolation (5-10 mutations). Designed sequences were synthesized, expressed in E. coli, and fluorescence was measured as relative brightness (GFP/mKate2 ratio normalized to WT). The bar chart shows that most METL-designed variants, particularly in the "observed" group, maintain measurable fluorescence, significantly outperforming baseline methods.

---

**Figure 6d — Random baseline variants brightness**  

**Goal:** establish a negative control to demonstrate that METL's design success is not due to chance.  
**How we did it:** we generated random variants matching the mutation counts and compositional constraints of METL-designed sequences. These random baseline variants were synthesized and measured under identical experimental conditions. The bar chart shows that randomly designed variants largely fail to maintain fluorescence, providing strong evidence that METL's success stems from learned sequence-function relationships rather than random exploration of sequence space.
