**Figure 6c — METL-designed variants brightness**

**Goal:** evaluate the functional success of protein sequences designed by METL under extreme low-data conditions (N=64 experimental training examples).  
**How we did it:** we fine-tuned METL (pretrained on 20M simulated examples) with only 64 experimental measurements, then used simulated annealing to design novel variants requiring mutation extrapolation (unobserved amino acids) and regime extrapolation (5-10 mutations). Designed sequences were synthesized, expressed in E. coli, and fluorescence was measured as relative brightness (GFP/mKate2 ratio normalized to WT). The bar chart shows that most METL-designed variants, particularly in the "observed" group, maintain measurable fluorescence, significantly outperforming baseline methods.

---

**Figure 6d — Random baseline variants brightness**  

**Goal:** establish a negative control to demonstrate that METL's design success is not due to chance.  
**How we did it:** we generated random variants matching the mutation counts and compositional constraints of METL-designed sequences. These random baseline variants were synthesized and measured under identical experimental conditions. The bar chart shows that randomly designed variants largely fail to maintain fluorescence, providing strong evidence that METL's success stems from learned sequence-function relationships rather than random exploration of sequence space.
