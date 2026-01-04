## New cross validation experiments

The previous FoIKS results were based on one 80-20 split. These days I have tried 4-fold cross validation (on the training set).

### Results Summary

The table below shows the 4-fold cross validation 

| Logic | desc_train | desc_test | synt_train | synt_test | comb_train | comb_test |
|--------|------------|-----------|------------|-----------|------------------|-----------------|
| ABV | 1.9 ± 1.1 | 2.0 ± 3.2 | 99.7 ± 0.1 | 70.7 ± 0.9 | 84.7 ± 1.0 | 68.0 ± 7.0 |
| ALIA | 3.4 ± 3.4 | -28.3 ± 23.0 | 98.6 ± 0.8 | 62.0 ± 7.3 | 78.9 ± 5.1 | 46.5 ± 18.1 |
| BV | 39.0 ± 3.6 | 7.4 ± 24.5 | 95.3 ± 1.0 | 32.0 ± 20.9 | 81.7 ± 2.0 | 31.2 ± 12.8 |
| QF_IDL | 69.8 ± 2.3 | 51.7 ± 3.8 | 92.1 ± 2.1 | 74.3 ± 14.0 | 91.1 ± 1.4 | 72.3 ± 7.3 |
| QF_LIA | 88.6 ± 0.8 | 85.6 ± 3.8 | 93.8 ± 1.5 | 86.6 ± 7.5 | 95.8 ± 0.3 | 90.7 ± 3.8 |
| QF_NRA | 63.4 ± 2.4 | 35.1 ± 5.1 | 56.0 ± 4.5 | 32.2 ± 20.5 | 76.9 ± 2.6 | 46.3 ± 4.4 |
| QF_SLIA | 86.6 ± 0.5 | 86.6 ± 1.5 | 96.7 ± 0.5 | 87.0 ± 3.7 | 96.0 ± 0.4 | 92.8 ± 2.6 |
| UFLIA | 3.5 ± 1.7 | -16.3 ± 20.2 | 98.8 ± 0.5 | 9.9 ± 36.0 | 80.3 ± 2.3 | 22.7 ± 35.2 |
| UFNIA | 1.1 ± 2.0 | -6.0 ± 10.4 | 70.9 ± 4.6 | 20.6 ± 9.0 | 55.1 ± 6.9 | 16.3 ± 19.0 |

It shows that the results do contain variability, especially for logics ALIA, BV, UFLIA, and UFNIA. 

From the new results, syntactic features alone is always better than description embedding alone, except for QF_NRA. The combined features outperform the syntactic alone in 4/9 cases, but are obviously worse in ABV, ALIA, and UFNIA.

### Analysis on low performance of description embeddings

Usually the SMT-LIB descriptions are identical within a family. I collected the familiy distribution statistics under [`data/features/family/summary`](../data/features/family/summary). Notably, in some logics, the number of families is very small or the family distribution is highly imbalanced. For example, in ABV and ALIA, they both only have 2 families, and one of the two accounts for more than 95% of all the benchmarks. One would not expect in such cases the description embedding can be useful.
In UFLIA and UFNIA, although there are a few families (12 and 9 respectively), each logic still has a dominant family that accounts for more than 50% of all the benchmarks. 

### Analysis on high variability in performance

The metrics used is PAR-2 VBS-SBS gap closed. This metric would be pretty sensitive to small performance variations when the VBS-SBS margin is small. The table below shows the dataset statistics. For example, for the full UFLIA dataset, the VBS solves 29 more instances than the SBS. Then, across test splits, the average VBS-SBS margin is ~6 instances, with larger margins for some splits and smaller for others. Consequently, solving 1 more or less instance can result in relatively large performance variations both within a single test split and across different splits.

| Logic | Size | #SBS Solved | #VBS Solved | VBS-SBS Solved Margin |
|-------|------|-------------|-------------|----------------------|
| ABV | 2487 | 837 | 971 | 134 |
| ALIA | 1537 | 298 | 390 | 92 |
| BV | 1040 | 952 | 1002 | 50 |
| QF_IDL | 1208 | 1047 | 1101 | 54 |
| QF_LIA | 4825 | 4514 | 4726 | 212 |
| QF_NRA | 3103 | 2812 | 2932 | 120 |
| QF_SLIA | 23722 | 23276 | 23700 | 424 |
| UFLIA | 2849 | 1628 | 1657 | 29 |
| UFNIA | 6279 | 3688 | 3759 | 71 |

We can see that the logics with larger performance variance (ALIA, BV, UFLIA, and UFNIA) all have relatively small VBS-SBS margins (< 100 instances). Also, for ALIA, the dataset may not be very sufficient because the VBS can only solve 390 instances. Previously, I set the logic filtering threshold to be at least 1000 instances in total. But for instances that can not be solved by the VBS, they are useless for training or testing.

## Plans

### LLM-generated descriptions

Since the native descriptions does not distinguish well between instances within a family, generating more informative descriptions with LLMs may help. Maybe agent framework is not necessary for now. I will try feeding LLM with necessary info (such as syntax features, variables by types, sample assertions, etc.). One thing worth working on is how to cluster assertions by similarity, and then sample assertions from clusters.

### Combine syntactic and description features

Currently, we simply concatenate the syntactic and description features, which is likely not the best way to leverage more information. Ideally, the approach should be robust to the inclusion of weakly informative features, so that performance does not degrade when such features are added.

One thing we can try is to use alternative base classifiers, such as XGBoost, which is better suited to handling heterogeneous feature types than our current SVM. Another thing is we can try *late fusion*, in which separate models are trained on syntactic and description features, respectively, and their predictions are combined using a weighted sum, i.e., $\text{pred} = \alpha \cdot \text{synt_model} + (1-\alpha) \cdot \text{desc_model}$, where $\alpha$ can be calibrated via cross-validation. 