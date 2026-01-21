# Fusion Methods Summary

```python
FUSION_METHODS = ["fixed_alpha", "confidence_weighted", "gated"]
```

All three methods operate at the **decision level** — they combine the pairwise classifier outputs (scores/decisions) after each model makes predictions but before the final solver ranking. This allows leveraging confidence/scores from each model's pairwise classifiers while keeping the models trained separately.

---

## 1. Fixed Alpha

Weighted average of decision scores with a tuned alpha parameter.

**Formula:**
```
decision_combined = α * score_synt + (1 - α) * score_desc
```

**Alpha tuning (nested cross-validation):**

1. **Outer loop** (e.g., 4 folds): Holds out 1 fold for testing
2. **Inner loop** (e.g., 3 folds on remaining training data): Tunes alpha
   - For each alpha candidate (default: 0.1, 0.2, ..., 0.9)
   - Train fusion model on inner train folds
   - Evaluate on inner validation fold
   - Track `gap_cls_par2` performance
3. **Select best alpha**: The alpha with highest mean `gap_cls_par2` across inner folds
4. The best alpha per fold is tracked (`best_alpha_mean` in results)

**How decision scores are calculated:**

1. **Get raw decision score:**
   - **SVM:** `classifier.decision_function()` → signed distance from hyperplane
   - **XGBoost:** `predict_proba()` → converted to log-odds: `log(p / (1-p))`
   - Prediction `{0, 1}` is mapped to `{-1, +1}`

2. **Standardize:**
   ```
   (decision - μ) / σ
   ```
   Where μ and σ are computed on training data for each pairwise classifier (i, j).

3. **Combine:**
   ```
   decision_combined = α * decision_synt + (1 - α) * decision_desc
   ```

---

## 2. Confidence Weighted

Dynamic weighting based on `|decision|` as confidence.

- Uses the absolute value of each model's decision score as a confidence measure
- Higher confidence → more weight in the fusion

**How weights are calculated:**

```python
# Confidence = absolute value of standardized decision
conf_synt = abs(decision_synt)
conf_desc = abs(decision_desc)

# Weight proportional to confidence
total_conf = conf_synt + conf_desc + 1e-9
weight_synt = conf_synt / total_conf
weight_desc = conf_desc / total_conf

# Combine
decision_combined = weight_synt * decision_synt + weight_desc * decision_desc
```

**Intuition:** If a classifier is "uncertain" (decision close to 0), it gets low weight. If confident (`|decision|` is large), it gets high weight. This is dynamic per instance and per pairwise comparison.

**Example flow for a single instance:**

```python
# For EACH pairwise comparison (i,j) in this instance:
for i in range(solver_size):
    for j in range(i + 1, solver_size):
        # Get decisions for THIS instance, THIS pair
        decision_synt = ...  # e.g., +0.8
        decision_desc = ...  # e.g., -0.2

        # Confidence = |decision|
        conf_synt = abs(decision_synt)  # 0.8
        conf_desc = abs(decision_desc)  # 0.2

        # Dynamic weights (like a "local alpha" for this comparison)
        weight_synt = 0.8 / (0.8 + 0.2)  # = 0.8
        weight_desc = 0.2 / (0.8 + 0.2)  # = 0.2

        # Combine
        decision_combined = 0.8 * (+0.8) + 0.2 * (-0.2)  # = 0.60
```

---

## 3. Gated

Learned meta-model that predicts the winner from decision scores.

- Trains a classifier on top of the two models' decision scores
- The meta-model learns which model to trust for different inputs

**Feature space:**
```
[decision_synt, decision_desc]
```

**Meta-model:** `LogisticRegression` with `StandardScaler` (one per solver pair)

**Output:** Probability that solver i beats solver j → used for **soft voting**:

```python
votes[i] += prob_i_wins
votes[j] += (1 - prob_i_wins)
```
