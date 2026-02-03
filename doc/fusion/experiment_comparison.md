# Experiment Set Comparison

## Overview

- **Experiment Set** (55779956): tune alpha on full training set and test on validation set, 0.05 alpha intervals

Metric: `test_gap_cls_par2_mean ± test_gap_cls_par2_std`

---

## Results Comparison

### All Methods: Baselines and Fusion Experiments

| Logic | desc_test | synt_test | comb_test | fusion_E3 | Best Method |
|-------|-----------|-----------|-----------|-----------|-------------|
| ABV | 2.0 ± 3.2 | 70.7 ± 0.9 | 68.0 ± 7.0 | **71.78 ± 1.01** | **Fusion E3** |
| ALIA | -28.3 ± 23.0 | 62.0 ± 7.3 | 46.5 ± 18.1 | **63.71 ± 6.37** | **Fusion E3** |
| BV | 7.4 ± 24.5 | **32.0 ± 20.9** | 31.2 ± 12.8 | 31.49 ± 20.30 | **Syntactic/Fusion E3** |
| QF_IDL | 51.7 ± 3.8 | 74.3 ± 14.0 | 72.3 ± 7.3 | **75.93 ± 13.80** | **Fusion E3** |
| QF_LIA | 85.6 ± 3.8 | 86.6 ± 7.5 | **90.7 ± 3.8** | **88.54 ± 6.11** | **Concat/Fusion E3** |
| QF_NRA | 35.1 ± 5.1 | 32.2 ± 20.5 | **46.3 ± 4.4** | **44.78 ± 14.49** | **Concat** |
| QF_SLIA | 86.6 ± 1.5 | 87.0 ± 3.7 | **92.8 ± 2.6** | N/A | **Concat** |
| UFLIA | -16.3 ± 20.2 | 9.9 ± 36.0 | **22.7 ± 35.2** | 9.91 ± 35.99 | **Concat** |
| UFNIA | -6.0 ± 10.4 | **20.6 ± 9.0** | 16.3 ± 19.0 | 20.55 ± 8.99 | **Syntactic/Fusion E3** |

