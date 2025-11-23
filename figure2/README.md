# Figure 2 — METL performance over experiment data size**

**Goal:** evaluate performance of METL against baseline methods in data-scarce protein engineering scenarios. Validating feasibility for adopting biophysics as prior in protein engineering

**How we did it:** we systematically varied training set sizes (from 8 to 16,384 samples) across 11 protein fitness datasets and measured test set performance using Spearman correlation. Methods included biophysics-based (METL-Global, METL-Local), evolutionary (ESM-2, EVE), and simple baselines (Linear, Rosetta). In our reproduction, we focus on the biophysics-based METL, which is purposed by the paper. Learning curves on log-log scales reveal how each method's inductive biases affect sample efficiency, with METL-Local excelling in extreme low-data regimes due to its protein-specific biophysical pretraining.

---

**Key Findings:**
- **Protein-specific patterns:** different proteins show varying optimal method preferences, highlighting the context-dependent value of different prior knowledge types
- **Sample efficiency:** METL frameworks maintain competitive performance within scarce dataset than general-purpose models when it comes to protein that is strongly related to biophysics features

The reproduction confirms the original paper's conclusion that biophysical priors provide critical advantages for practical protein engineering where experimental data is severely limited.

-note the paper compare different models with METL, we only write the code for the METL, as all those other model are other papers' result.

# key problem when encoutered:
-There is a lot to be learned, about how to run this, and we spent a lot of time trying to learn how to use and do the training, the initial part is very hard, a lot things is different, but afterward things get easier, as the core layout is all the data, it is basically extra data processing when we fully understand how to train the model. Several issues occurred during training which are discussed during presentation. For example, we initially are confuse about the naming of the models so we use a pre-tuned model and let it predict the spearman which results in insanely high spearman coefficient. Or about we forgetting to normalized the data, and no pre-stopping at small N due to we set the threshold for early stopping too high, so we train the full 250 epochs on tiny data and overfit like crazy. And the ridiculous training time, as we need so many replicates for each and we need to un freeze the whole model after training head, this is why in our extension we only unfreeze the top layer, it is just way too long. We use the exact same set up, the learning rate, weight decay , and everything, we print the progress out nicely so we can track it, as initially we thought our pc breaks. So we print out our training process nicely and MSE for us to track if it is properly going down. Also, after we written the code, we try to run it on multiple computers, as one pc can only run one at a time. I have a habit to make sure everything thing is the same version when I setup my local environment, so I will reinstall things like NumPy or torch to make sure they are the exact same version as the authors, but when our other member setup their environment, so some weird reasons their SciPy don’t work, so we just code our own spearman coefficient function as it is easier than to re sorting everything. So some of our code have our own implementation of spearman.

There are also things not mentioned, like the need of batches, we understand the idea but initially we did not learn how to do this so we did not do it. And our RAM blow up then we try to fit everything all at once, so we have to do batch. But we do not know what size to choose, so we pick a ‘sensible’ one base on memory ∝ (batch_size × sequence_length² × number_of_heads), I only have 16GB RAM, I do not want it to use all as I want to be able to still use my pc during training, and this properly results in the long time. So we later decide to increase our batch size from 128 to 256 for faster training.





