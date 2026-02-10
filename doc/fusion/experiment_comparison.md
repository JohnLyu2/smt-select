# Experiment Set Comparison

## Overview

- **Fusion Validation** (experiment 56302061): tune alpha on full training set and test on validation set, 0.05 alpha intervals
- **Fusion train/test** (experiment 56316328): use the found alpha to fuse models trained on train set and evaluate on test set

Metric: `test_gap_cls_par2_mean ± test_gap_cls_par2_std`

---

## Results Comparison

### All Methods: Baselines and Fusion Experiments

| Logic | desc_test | synt_test | comb_test | fusion_validation | best_alpha |
|-------|-----------|-----------|-----------|-----------|------------|
| ABV | 2.0 ± 3.2 | **70.7 ± 0.9** | 68.0 ± 7.0 | **70.70 ± 0.88** | 1.0 | 
| ALIA | -28.3 ± 23.0 | 62.0 ± 7.3 | 46.5 ± 18.1 | **63.14 ± 5.03** | 0.95 | 
| BV | 7.4 ± 24.5 | **32.0 ± 20.9** | 31.2 ± 12.8 | **32.04 ± 20.94** | 1.0 | 
| QF_IDL | 51.7 ± 3.8 | 74.3 ± 14.0 | 72.3 ± 7.3 | **77.96 ± 11.02** | 0.95 | 
| QF_LIA | 85.6 ± 3.8 | 86.6 ± 7.5 | **90.7 ± 3.8** | 88.28 ± 3.18 | 0.65 |
| QF_NRA | 35.1 ± 5.1 | 32.2 ± 20.5 | **46.3 ± 4.4** | 45.35 ± 12.71 | 0.8 | 
| QF_SLIA | 86.6 ± 1.5 | 87.0 ± 3.7 | **92.8 ± 2.6** | 91.93 ± 2.71 | 0.95 | 
| UFLIA | -16.3 ± 20.2 | 9.9 ± 36.0 | **22.7 ± 35.2** | 11.64 ± 27.80 | 0.95 | 
| UFNIA | -6.0 ± 10.4 | **20.6 ± 9.0** | 16.3 ± 19.0 | **20.55 ± 9.00** | 1.0 | 
### Fusion Train/Test Gap (train_eval_fusion)

| Logic | fusion_train | fusion_test |
|-------|--------------|-------------|
| ABV | 0.997 | 0.9239 |
| ALIA | 0.983 | 0.621 |
| BV | 0.9304 | 0.1195 |
| QF_IDL | 0.9126 | 0.6284 |
| QF_LIA | 0.907 | 0.9431 |
| QF_NRA | 0.6818 | 0.6891 |
| QF_SLIA | 0.9796 | 0.9643 |
| UFLIA | 0.7734 | 0.597 |
| UFNIA | 0.7134 | 0.3911 |

