# Path Embedding Experiment Results

## Overview

This experiment compares **path embeddings** (embeddings of SMT-LIB file paths like `"QF_LIA/family/benchmark.smt2"`) against **description embeddings** from the baseline experiment.

**Hypothesis**: Path strings contain semantic information (logic, family, benchmark name) that may be more distinctive than descriptions, especially for logics where descriptions are identical within families.

## Experimental Setup

- **Model**: sentence-transformers/all-mpnet-base-v2 (768-dim embeddings)
- **Cross-validation**: 4-fold CV
- **Evaluation metric**: gap_cls_par2 (% gap closed between SBS and VBS on PAR-2 score)

### Experiments Run

1. **path_emb**: Path embeddings only (768-dim)
2. **combined_path_synt**: Path embeddings + Syntactic features (~858-dim concatenated)

### Baseline Comparison (from experiment0103.md)

- **desc**: Description embeddings only (768-dim)
- **synt**: Syntactic features only (~90-dim)
- **comb**: Description embeddings + Syntactic features (~858-dim)

## Results Summary

### Key Findings from ABV Logic

From the summary.json files, for **ABV** logic:

**Path Embeddings Only (path_emb)**:
- Test gap_cls_par2: **60.6% ± 4.1%**

**Path + Syntactic (combined_path_synt)**:
- Test gap_cls_par2: **64.9% ± 9.7%**

**Baseline Comparison (from experiment0103.md)**:
- Description only (desc_test): **2.0% ± 3.2%**
- Syntactic only (synt_test): **70.7% ± 0.9%**
- Description + Syntactic (comb_test): **68.0% ± 7.0%**

### Analysis for ABV

**Path vs Description**:
- Path embeddings: **60.6%** vs Description embeddings: **2.0%**
- **Improvement: +58.6 percentage points** ✅

This is a massive improvement! Path embeddings are significantly more informative than description embeddings for ABV.

**Combined Features**:
- Path + Syntactic: **64.9%** vs Description + Syntactic: **68.0%**
- Slightly lower (-3.1%), but both are comparable
- Path + Syntactic: **64.9%** vs Syntactic only: **70.7%**
- Adding path embeddings slightly degrades syntactic-only performance

**Why Path Embeddings Work Better for ABV**:

From experiment0103.md analysis:
- ABV has only **2 families**, with one accounting for >95% of benchmarks
- **Descriptions are identical within each family**
- Path strings contain **instance-level information** (benchmark names) that descriptions lack
- Example paths: `"ABV/UltimateAutomizerSvcomp2019/alternating_list_14.smt2"` vs `"ABV/UltimateAutomizerSvcomp2019/alternating_list_15.smt2"`

## Next Steps for Complete Analysis

To generate the full comparison table for all 9 logics, run:

```bash
# Submit the analysis job
sbatch scripts/jobs/analyze_results.sh

# Or run directly
cd /lustre06/project/6001884/jchen688/SMT_AS_LM
source /home/jchen688/projects/def-vganesh/jchen688/z3_env/bin/activate
python scripts/compare_path_vs_baseline.py > doc/path_vs_baseline_comparison.md
```

This will generate:
1. Complete comparison table for all 9 logics
2. Path vs Description improvements for each logic
3. Combined feature comparisons
4. Summary statistics

## Expected Results for Other Logics

Based on the hypothesis and baseline analysis:

**Logics where path embeddings should significantly outperform descriptions**:
- **ABV**: ✅ Confirmed (+58.6%)
- **ALIA**: Expected (2 families, >95% imbalance, desc_test = -28.3%)
- **UFLIA**: Expected (dominant family >50%, desc_test = -16.3%)
- **UFNIA**: Expected (dominant family >50%, desc_test = -6.0%)

**Logics where both may perform similarly**:
- **QF_LIA**, **QF_SLIA**: Descriptions already perform well (85-86%)
- **QF_IDL**: Moderate description performance (51.7%)

**Logics to watch**:
- **QF_NRA**: Only logic where desc > synt in baseline (desc_test = 35.1% vs synt_test = 32.2%)
- **BV**: High variability, small VBS-SBS margin

## Implementation Details

### Path Encoding with Error Handling

The `path_encoder.py` was enhanced to handle missing `smtlib_path` fields:

```python
# Use placeholder path if missing or empty
if not smtlib_path or not smtlib_path.strip():
    logic = benchmark.get("logic", "unknown")
    family = benchmark.get("family") or "unknown"
    benchmark_name = benchmark.get("benchmark_name", "unknown.smt2")
    # Create placeholder path for embedding
    smtlib_path = f"{logic}/{family}/{benchmark_name}"
```

This fixed the QF_NRA encoding issue (1 benchmark with missing path).

### Files Created

1. **`src/path_encoder.py`**: Path embedding module (reuses `desc_encoder.py` infrastructure)
2. **`scripts/encode_all_paths.py`**: Batch encoding for all 9 logics
3. **`scripts/jobs/encode_paths.sh`**: SLURM job for encoding
4. **`scripts/jobs/run_cv_path_emb.sh`**: CV job for path embeddings only
5. **`scripts/jobs/run_cv_combined_path_synt.sh`**: CV job for combined features
6. **`scripts/compare_path_vs_baseline.py`**: Analysis script
7. **`data/features/path_emb/all_mpnet_base_v2/*.csv`**: Path embedding CSVs (9 logics)
8. **`data/cv_results/path_emb/`**: CV results for path embeddings only
9. **`data/cv_results/combined_path_synt/`**: CV results for combined features

## Conclusions

Based on the ABV results:

1. **Path embeddings significantly outperform description embeddings** when descriptions lack instance-level distinctiveness (+58.6% for ABV)

2. **Path embeddings capture meaningful information** about benchmark characteristics through:
   - Logic type (e.g., "ABV", "QF_LIA")
   - Family name (e.g., "UltimateAutomizerSvcomp2019")
   - Benchmark name (e.g., "alternating_list_14.smt2")

3. **Combined features (path + syntactic) are competitive** but syntactic features alone still perform best for ABV

4. **This validates the hypothesis** that path strings provide valuable semantic information for algorithm selection when descriptions are not sufficiently diverse

## Future Work

1. **Complete the analysis** for all 9 logics using `compare_path_vs_baseline.py`
2. **Investigate why syntactic features outperform both embeddings** for most logics
3. **Try late fusion** instead of early concatenation for combining features
4. **Experiment with path augmentation** (e.g., adding file size, number of variables to path string)
5. **Fine-tune embedding model** on path strings for even better performance
