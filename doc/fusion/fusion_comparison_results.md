# Fusion Methods Comparison Results

Comparison of **decision-level fusion** methods against **baseline** methods across all logics.

Metric: **Gap Closed (PAR-2)** — higher is better, represents (AS - SBS) / (VBS - SBS).

> Note: Gated method currently excluded due to Compute Canada Narval failures.

---

## Test Set Performance (Gap Closed PAR-2 %)

| Logic | desc | synt | comb | fixed_alpha | conf_weighted | Best |
|-------|------|------|------|-------------|---------------|------|
| ABV | 2.0 ± 3.2 | 70.7 ± 0.9 | 68.0 ± 7.0 | -65.9 ± 26.9 | -121.7 ± 46.6 | **synt** |
| ALIA | -28.3 ± 23.0 | 62.0 ± 7.3 | 46.5 ± 18.1 | 51.7 ± 12.5 | 51.7 ± 12.5 | **synt** |
| BV | 7.4 ± 24.5 | 32.0 ± 20.9 | 31.2 ± 12.8 | 23.7 ± 30.1 | 6.7 ± 32.5 | **synt** |
| QF_IDL | 51.7 ± 3.8 | 74.3 ± 14.0 | 72.3 ± 7.3 | -46.5 ± 57.2 | -105.2 ± 61.1 | **synt** |
| QF_LIA | 85.6 ± 3.8 | 86.6 ± 7.5 | 90.7 ± 3.8 | 86.9 ± 4.0 | 87.4 ± 3.7 | **comb** |
| QF_NRA | 35.1 ± 5.1 | 32.2 ± 20.5 | 46.3 ± 4.4 | 28.5 ± 10.2 | 13.5 ± 14.5 | **comb** |
| QF_SLIA | 86.6 ± 1.5 | 87.0 ± 3.7 | 92.8 ± 2.6 | 89.1 ± 1.7 | 70.2 ± 9.8 | **comb** |
| UFLIA | -16.3 ± 20.2 | 9.9 ± 36.0 | 22.7 ± 35.2 | -1360.8 ± 205.5 | -1736.0 ± 205.2 | **comb** |
| UFNIA | -6.0 ± 10.4 | 20.6 ± 9.0 | 16.3 ± 19.0 | -2254.6 ± 688.5 | -2754.7 ± 727.7 | **synt** |

---

## Best Alpha Values (fixed_alpha fusion)

| Logic | Best Alpha | Interpretation |
|-------|------------|----------------|
| ABV | 0.18 | Favors description |
| ALIA | 0.70 | Favors syntactic |
| BV | 0.77 | Favors syntactic |
| QF_IDL | 0.35 | Slightly favors description |
| QF_LIA | 0.53 | Balanced |
| QF_NRA | 0.15 | Favors description |
| QF_SLIA | 0.38 | Slightly favors description |
| UFLIA | 0.10 | Strongly favors description |
| UFNIA | 0.30 | Favors description |

---

## Insights

1. **Decision-level fusion fails to outperform baselines** — In 0/9 logics, fusion methods beat the best baseline. The best method is always either `synt` (5/9) or `comb` (4/9).

2. **Fusion performs catastrophically on UFLIA and UFNIA** — Gap values indicate severe degradation, far worse than any baseline.

3. **Fixed_alpha consistently beats confidence_weighted** — In 8/9 logics, fixed_alpha performs better or equal.

**Conclusion:** Decision-level fusion as currently implemented does not provide complementary benefits. Feature-level concatenation (`comb`) or using syntactic features alone outperforms fusion in all tested logics.
