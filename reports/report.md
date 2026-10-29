# Phishing Website Detection Using From-Scratch Logistic Regression

*7th-semester Computer Engineering — ML Project-Based Learning.*
Every number below is produced by `python -m src.eda` and `python -m src.train` and is stored in
`reports/eda_results.json` and `reports/results.json`. Figures are in `reports/figures/`
(Logistic-Regression-only evaluation figures: `python -m src.lr_only_figures`).

## 1. Abstract

We classify websites as phishing or legitimate from 30 pre-extracted indicators of the UCI Phishing Websites dataset,
using **logistic regression implemented from scratch in NumPy** and trained by gradient descent with binary
cross-entropy and L2 regularisation.

An audit found that 47% of the 11,055 rows are exact duplicates. We therefore evaluated on 5,721 unique feature vectors,
split 70/15/15 before any modelling. All choices were made on validation data only. On the untouched test set
(859 rows):

| Model | F1 [95% bootstrap CI] | ROC-AUC [95% CI] |
|---|---|---|
| Logistic regression (30 inputs) | 0.937 [0.920, 0.952] | 0.988 [0.983, 0.992] |
| Logistic regression (8 inputs, app) | 0.932 [0.914, 0.948] | 0.987 [0.982, 0.991] |
| Best single feature (baseline) | 0.887 [0.865, 0.910] | 0.897 [0.878, 0.918] |

The model clearly beats the best single-feature baseline (paired ΔF1 = +0.050 [+0.029, +0.070]) and is statistically
indistinguishable from a scikit-learn reference trained on the same objective. Two features carry most of the signal:
HTTPS certificate state and anchor-link origin. Several features behave differently from their documentation, and the
data are from around 2012, so the results should not be read as a guarantee about current websites.

## 2. Introduction and motivation

Phishing sites imitate trusted brands to steal credentials. Blacklists react only after a site is reported. Classifiers
built on URL, page and domain indicators can flag new sites earlier. The goal of this project is not a record score.
It is a transparent, reproducible experiment in which the algorithm is written and verified by hand and every
number can be traced back to the code that produced it.

## 3. Problem definition

This is binary classification. Input: 30 discrete indicators x ∈ {−1, 0, 1}³⁰. Output: an estimated probability
P(phishing | x) and a class at a validation-chosen threshold.

- **Positive class:** phishing = 1; legitimate = 0.
- **Primary metrics:** F1 and ROC-AUC for phishing.
- **Also reported:** accuracy, precision, recall, macro-F1, PR-AUC, log-loss and confusion counts.

## 4. Dataset source and license

- **Dataset:** UCI Machine Learning Repository, *Phishing Websites* (id 327), DOI 10.24432/C51W2X, licensed
  CC BY 4.0. Creators: Rami Mohammad and Lee McCluskey.
- **Intro paper:** Mohammad, Thabtah & McCluskey (2012), *An assessment of features related to phishing websites using
  an automated technique*, ICITST.
- **Retrieval:** `ucimlrepo` retrieved the metadata, but its data download failed on a local Python certificate-store
  error. The same `data.csv` was downloaded with `curl`, with TLS verification on. Its values are identical to the ARFF
  file in the official zip. SHA-256 `35dd9aa6…59c2`.

## 5. Feature descriptions

The features fall into four groups:

| Group | Count | Examples |
|---|---|---|
| URL / address bar | 12 | IP address, URL length, '@', prefix–suffix dash, sub-domains, HTTPS certificate state |
| Abnormal / page content | 6 | external objects, anchor targets, links in tags, form handler |
| HTML / JavaScript | 5 | redirects, status-bar tricks, disabled right-click, pop-ups, iframes |
| Domain / reputation | 7 | domain age, DNS record, traffic rank, PageRank, Google index |

Full descriptions, value meanings and observed statistics are in `reports/features.md`.

The documentation never states the numeric codes. The −1 = phishing-like convention is supported by internal checks
(§6) but is not proven. Three features have documented ambiguities:

- `redirect` is coded {0, 1}, and the meaning of each code is undocumented.
- `prefix_suffix`: all 4,898 raw phishing rows have −1, which is also the majority code (87%).
- `domain_registration_length` is associated with the label in the direction opposite to its documented rule.

## 6. Dataset inspection and duplicate findings (`reports/dataset_inspection.md`)

| Quantity | Value |
|---|---|
| Rows × columns | 11,055 × 31, all integers, no missing values, all values in {−1, 0, 1} |
| Raw class balance | −1: 4,898 (44.3%), +1: 6,157 (55.7%) |
| Exact duplicate rows | 5,206 |
| Unique feature vectors | 5,785 (largest repeat group: 25 rows) |
| Vectors with conflicting labels | 64 (357 raw rows) |
| Retained after policy | **5,721**: 2,955 phishing (51.7%), 2,766 legitimate (48.3%) |

**Policy (approved):** collapse exact duplicates, drop every vector that appears with both labels, then split.

- This changes the population from *collected websites* to *unique feature vectors*.
- Coarse indicators let distinct sites collide, so duplicates are not necessarily errors.

**Label direction check:**

- In all 9 features whose documented "phishing" condition is a rare anomaly, −1 is the rare code.
- The two strongest predictors both co-occur −1 with target −1.
- Together these support −1 → phishing.

**Leakage screening:**

- No column equals the target or its negation.
- The strongest single feature, `sslfinal_state`, reaches validation ROC-AUC 0.898. That is strong, but it is a known
  phishing signal and is far from 1.

## 7. Exploratory analysis, preprocessing and split strategy

### Exploratory analysis (`notebooks/eda.ipynb`, figures 01–08)

Label-dependent statistics use the training partition only.

- **Two features dominate:**
  - `sslfinal_state`: Cramér's V 0.78, validation AUC 0.898.
  - `url_of_anchor`: Cramér's V 0.72, validation AUC 0.858.
- **9 features are no better than chance on their own:** their bootstrap AUC intervals include 0.5.
- **`sslfinal_state`, `having_sub_domain` and `web_traffic` are non-monotone in their codes.** Code 0 is riskier than −1.
- **Two clusters of strongly inter-associated features are barely related to the label.** The clearest example is
  `favicon`–`popupwindow`, with Cramér's V 0.93.
- **Duplicates would have overlapped across a raw split.** Splitting the raw file puts an exact twin in training for
  63.6% of validation rows. A pure look-up model is 97.3% accurate on those rows.

### Split

- Proportions are 70/15/15, stratified, with seed 42. The split is made after cleaning.
- Partition sizes are **4,004 / 858 / 859**, with phishing rates 0.517 / 0.516 / 0.517.
- Indices are saved to `data/processed/splits.json`.

### Encoding (`models/preprocessing.json`)

Both encoders are fitted on train.

- **Raw:** the 30 codes as numbers. Compared against one-hot and rejected for the final model, because a linear model
  treats −1 → 0 → 1 as evenly spaced and several features are non-monotone.
- **One-hot:** 38 indicator columns. The reference level of every feature is its highest code (1).

No scaling is used: the features are bounded codes and one-hot columns are already 0/1. Inputs with unseen codes,
missing columns or missing values are rejected.

### Baselines (validation, threshold 0.5)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Majority | 0.516 | 0.681 | 0.500 |
| Best single feature (`sslfinal_state`) | 0.888 | 0.889 | 0.898 |
| sklearn LR reference, raw codes | 0.930 | 0.931 | 0.978 |
| sklearn LR reference, one-hot | 0.945 | 0.947 | 0.988 |

The majority baseline shows why accuracy alone is misleading: F1 0.68 and recall 1.0 with zero skill.

## 8. Algorithm design

**Logistic regression** (`src/logistic_regression.py`):

- Model: p = σ(w·x + b).
- Objective: J = Σ sᵢℓᵢ/Σ sᵢ + (λ/2)‖w‖², with ℓ = log(1 + eᶻ) − yz and optional class weights sᵢ.
- Gradients: ∇w = Xᵀ(s ⊙ (p − y))/Σs + λw and ∂b = Σ s(p − y)/Σs. The bias is not regularised.
- Derivation: ∂ℓ/∂p = (p − y)/(p(1 − p)), ∂p/∂z = p(1 − p) and ∂z/∂w = x, so ∂ℓ/∂w = (p − y)x.
- Training: batch gradient descent (or mini-batch) from zero initialisation. Stops when |ΔJ| < 1e-9.
- Stable sigmoid computed piece-wise; the loss uses `logaddexp`, so it never takes log(0).
- The step size is set relative to the smoothness constant L = ¼λ_max(X̃ᵀX̃/n) + λ, since descent is guaranteed
  for η < 2/L.

## 9. Implementation details and verification

| Check | Result |
|---|---|
| Gradient vs finite differences (with and without L2 and class weights) | agree to rtol 1e-5 |
| LR weights vs scikit-learn on synthetic data, C = 1/(λn) | max difference < 1e-4 |
| LR on the real data vs scikit-learn (same objective) | max validation probability difference 0.016; 99.8% label agreement |
| Save → load → predict | identical outputs |
| Streamlit app (`streamlit.testing`) | renders 8 questions and no sidebar; rejects an empty form; shows gauge, verdict and main reason |
| Quick-model contributions | sum exactly to logit(p) − baseline logit |
| Zone thresholds | meet their recall / precision targets on synthetic data |

**Efficiency.** The selected model trains in about 14 s (27,079 iterations, peak traced memory ≈ 1.9 MB) and scores
the 858 validation rows in under 0.1 ms.

## 10. Experimental methodology

- **Selection criterion:** validation log-loss (a proper scoring rule, appropriate because the app shows probabilities).
- **Thresholds:** chosen to maximise validation F1, with ties going to the threshold nearest 0.5.
- **Grid:** encoding {raw, one-hot} × λ {0, 1e-4, 1e-3, 1e-2, 1e-1} × η {0.5, 1, 1.9}/L, with up to 30,000
  iterations. Mini-batch and class-weighted variants are reported as diagnostics.
- **Cross-validation:** stratified 5-fold on the training partition only.
- **Test evaluation:** one evaluation of the frozen saved model. 95% percentile bootstrap intervals (1,000 resamples)
  and **paired** bootstrap differences. Overlapping separate intervals are not treated as a significance test.

## 11. Evaluation results

**Selected configuration:** one-hot, λ = 1e-4, η = 1/L ≈ 0.546, converged after 27,079 iterations, threshold 0.35.

**Test set** (859 rows; positive = phishing):

| Model | Acc. | Prec. | Recall | F1 | Macro-F1 | ROC-AUC | PR-AUC | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Custom LR** | 0.934 | 0.918 | 0.957 | **0.937** | 0.933 | **0.988** | 0.989 | 425 | 38 | 19 | 377 |
| sklearn LR (ref) | 0.934 | 0.918 | 0.957 | 0.937 | 0.933 | 0.988 | 0.989 | 425 | 38 | 19 | 377 |
| Best single feature | 0.885 | 0.897 | 0.878 | 0.887 | 0.885 | 0.897 | 0.874 | 390 | 45 | 54 | 370 |
| Majority | 0.517 | 0.517 | 1.000 | 0.681 | 0.341 | 0.500 | 0.517 | 444 | 415 | 0 | 0 |

Custom LR with 95% bootstrap intervals: F1 0.937 [0.920, 0.952], ROC-AUC 0.988 [0.983, 0.992],
PR-AUC 0.989 [0.984, 0.993].

**Paired bootstrap comparisons on test** (difference A − B with 95% interval):

| Comparison | ΔF1 | ΔROC-AUC | Verdict |
|---|---|---|---|
| LR − best single feature | +0.050 [+0.029, +0.070] | +0.090 [+0.071, +0.108] | clear improvement |
| Custom LR − sklearn LR | 0.000 | 0.000 | no meaningful difference |

**Cross-validation** (5 folds on train, mean ± sd):

| Model | ROC-AUC | F1 |
|---|---|---|
| Custom LR | 0.990 ± 0.003 | 0.949 ± 0.006 |
| Best single feature | 0.897 ± 0.011 | 0.887 ± 0.011 |
| Majority | 0.500 ± 0.000 | 0.681 ± 0.001 |

**Reading the results.**

- On test the model catches 425 of 444 phishing sites (recall 0.957) with 38 false alarms among 415 legitimate sites.
- The threshold of 0.35 was tuned for F1 on validation. On test it traded precision for recall more than it did on
  validation: validation F1 was 0.953, test F1 0.937. Validation-chosen thresholds do not transfer perfectly.
- ROC-AUC measures ranking over all thresholds; precision, recall and F1 depend on the chosen threshold, so both are
  reported.
- Figures: `lr_05` (ROC and precision–recall), `lr_06` (confusion matrices), `lr_04` (threshold curve).

**Duplicate diagnostic.** We retrained with the same settings on a raw (non-deduplicated) split:

| Model | Raw-split validation accuracy | Clean validation accuracy |
|---|---|---|
| LR | 0.930 | 0.951 |

So, on this dataset (LR validation accuracy 0.930 vs 0.951), duplicates did **not** inflate scores, even though 63.6%
of raw validation rows have a training twin. We report this as a negative finding. Deduplication remains the
methodologically correct choice, because it removes the possibility of inflation rather than relying on it being small.

## 12. Overfitting analysis (`lr_01`–`lr_03`, `lr_08`)

- **Train vs validation.** At the selected setting, F1 is 0.952 on train and 0.953 on validation, and log-loss is
  0.123 vs 0.139: the model is not overfitting.
- **Regularisation.** With no penalty (λ = 0) validation log-loss is 0.139; at λ = 1e-4 it is 0.139 (selected, the
  simplest within rounding). Strong L2 under-fits: at λ = 0.1 train and validation log-loss rise together to 0.396.
- **Encoding.** Raw −1/0/1 codes under-fit at every λ (best validation log-loss 0.191 vs 0.139 for one-hot), because a
  linear model cannot represent non-monotone codes.
- **Learning rate.** Rates below 2/L ≈ 1.09 decrease the loss monotonically; 2 and 4 oscillate at the start.
- **Learning curve.** Validation ROC-AUC rises from 0.980 (399 rows) to 0.988 (4,004 rows) and is flattening, with a
  small, stable train–validation gap: more data would help only a little.

| Setting | Train log-loss | Validation log-loss | Diagnosis |
|---|---|---|---|
| One-hot, λ = 1e-4 (selected) | 0.123 | 0.139 | good fit |
| One-hot, λ = 0.1 | 0.396 | 0.396 | under-fitting (too much regularisation) |
| Raw codes, λ = 0 | 0.166 | 0.192 | under-fitting (wrong representation) |

## 13. Feature selection and interpretation

**Feature selection** (training-only selection, validation evaluation):

- **Near-constant filter.** The dominant-level share is used instead of raw variance, because raw variance is not
  comparable when `redirect` is coded {0, 1}. τ = 0.95 removes only `rightclick`, with no loss. τ = 0.85 removes 7
  features and costs 0.003 AUC.
- **Mutual-information top-k:** k = 3 reaches validation AUC 0.965, k = 8 reaches 0.985 and k = 15 reaches 0.987,
  against 0.988 with all 30 (`lr_09`).
- **Ablations (validation AUC change, `lr_10`):**

| Ablation | Effect |
|---|---|
| Drop the HTML/JS cluster (7 features) | +0.001 |
| Drop `prefix_suffix` | −0.003 |
| Drop `sslfinal_state` | −0.018 |
| Drop `sslfinal_state` and `url_of_anchor` | −0.071 |

The deterministic `prefix_suffix` pattern helps slightly but is not what drives the results.

**Interpretation:**

- **Coefficients** (one-hot, relative to code 1), largest:

| Level | Coefficient |
|---|---|
| `url_of_anchor=-1` | +6.12 |
| `sslfinal_state=0` | +5.42 |
| `prefix_suffix=-1` | +4.68 |
| `sslfinal_state=-1` | +3.46 |

  - These are log-odds associations in this dataset, conditional on the other features. They are not causal effects.
  - Because the reference level is the legitimate-like code, the intercept is strongly negative (−11.7).
  - Coefficients of 9 weak or redundant levels (e.g. `popupwindow=-1`, `iframe=-1`) change sign across CV folds, so they
    should not be interpreted.
  - L2 shrinks ‖w‖ from 13.2 (λ = 0) to 1.4 (λ = 0.1).
- **Permutation importance** on validation (ROC-AUC drop, `lr_07`): `url_of_anchor` 0.077, `sslfinal_state` 0.055,
  `prefix_suffix` 0.016, `web_traffic` 0.006. Correlated features share credit when one is shuffled.
- **Practical meaning:**
  - **`sslfinal_state`:** sites without valid HTTPS from a trusted, aged certificate.
  - **`url_of_anchor`:** pages whose links mostly point elsewhere or nowhere.
  - **`prefix_suffix`:** domains with a dash.

  These match how phishing kits are typically built, but the data cannot show cause.

### Quick 8-input model for the app (`python -m src.train --stage quick`)

The 30-feature form was too long for a demo, so the app uses a smaller logistic regression.

- **Features.** The top 8 by mutual information, computed on the **training partition only**: `sslfinal_state`,
  `url_of_anchor`, `prefix_suffix`, `having_sub_domain`, `web_traffic`, `sfh`, `links_in_tags`, `request_url`. This is
  the same list as the `mi_top_8` experiment above; the stage checks that they match.
- **Training.** Same 4,004 training rows and the same grid as the full model, selected by validation log-loss:
  one-hot (14 inputs), λ = 1e-4, 23,325 iterations, threshold 0.28.
- **Why 8 (the measured trade-off).** Test set, one evaluation, 95% bootstrap intervals:

| Model | Inputs | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Quick LR | 8 | 0.28 | 0.927 | 0.896 | 0.971 | 0.932 [0.914, 0.948] | 0.987 [0.982, 0.991] |
| Full LR | 30 | 0.35 | 0.934 | 0.918 | 0.957 | 0.937 [0.920, 0.952] | 0.988 [0.983, 0.992] |

  Paired bootstrap (8 inputs minus 30): ΔF1 −0.005 [−0.019, +0.006] — *small or uncertain difference*;
  ΔROC-AUC −0.001 [−0.003, +0.002] — *no meaningful difference*. The model gets almost nothing from the other 22
  features, so 8 inputs cut the form by 73% at no measurable cost.
- **Three-level verdict.** Two cut-offs, chosen on **validation only** with a 98% target:
  - *low* = the largest threshold at which validation phishing recall is still ≥ 98% (at most 2% of phishing sites are
    called Safe);
  - *high* = the smallest threshold at which validation precision is ≥ 98% (at least 98% of "Likely phishing" sites
    are phishing).
  - Safe: p < low; Suspicious: low ≤ p < high; Likely phishing: p ≥ high.
  - Cut-offs: low 0.17, high 0.92. Both are saved in the model file and in `results.json` (`quick_models.zones`).
- **Zones on the test set** (859 sites):

| Zone | Sites | Share | Phishing rate |
|---|---|---|---|
| Safe | 337 | 39.2% | 2.1% |
| Suspicious | 162 | 18.9% | 50.0% |
| Likely phishing | 360 | 41.9% | 98.9% |

  The zones behave as designed on unseen data: *Safe* and *Likely phishing* are reliable, and the uncertain fifth of
  sites is set aside as *Suspicious*.
- **"Why this result?" contributions.** Each answer's contribution is Σ_j w_j (x_j − x̄_j) over that feature's
  one-hot columns, where x̄_j is the training mean. The contributions add up exactly to logit(p) minus the logit of an
  average training site (0.69). Positive values push towards phishing (red), negative towards legitimate (green).
  Centring is needed because with one-hot coding the reference answer has x_j = 0; the uncentred w_j·x_j would show
  every legitimate-like answer as zero, so nothing could ever explain a "Safe" result. The "Main reason" line names the
  largest contribution on the side of the decision.

## 14. Limitations

1. **Age and reproducibility.** The data are from around 2012. Several inputs relied on services that are gone (Alexa
   rank, public PageRank), and the feature thresholds cannot be re-derived. Nothing here demonstrates performance on
   current websites.
2. **Population.** Evaluation is on unique feature vectors after deduplication. The class balance and the probabilities
   are not representative of real traffic, so precision in deployment would be lower if phishing is rarer.
3. **Undocumented coding.** The label direction and three feature codings rest on internal evidence, not documentation.
4. **Single split.** Test intervals are from one 859-row test set. The validation-chosen threshold did not transfer
   perfectly.
5. **No live feature extraction.** The app accepts pre-extracted features only (the 8 inputs of the quick model).
   The app itself shows no limitations text; they are documented here.
6. **Linear model.** Logistic regression can only add up evidence feature by feature; it cannot learn interactions
   between features unless they are engineered by hand.

## 15. Future scope

- A documented, independently validated feature extractor for live URLs, run in a sandbox.
- Evaluation on newer phishing corpora, with a time-based split to measure drift.
- Calibration (Platt/isotonic) on a separate calibration split.
- Cost-sensitive thresholds chosen from real costs of false alarms and misses.
- Hand-crafted interaction features, or a non-linear model, to capture combinations of indicators.

## 16. Conclusion and references

On the deduplicated UCI phishing data, a from-scratch logistic regression gives:

- test F1 0.937 and ROC-AUC 0.988, clearly better than the best single-feature baseline (ΔF1 +0.050);
- results indistinguishable from a scikit-learn reference, which supports the correctness of the implementation;
- an 8-input version for the app with no measurable loss (ΔF1 −0.005, uncertain).

Most of the signal comes from certificate and link-origin indicators. The main methodological lessons:

- audit duplicates before splitting;
- choose everything on validation data;
- report paired uncertainty;
- state the coding ambiguities and the dataset's age plainly.

**References**

1. Mohammad, R., Thabtah, F., McCluskey, L. (2012). An assessment of features related to phishing websites using an
   automated technique. *ICITST 2012*.
2. Mohammad, R., McCluskey, L. (2012). Phishing Websites [Dataset]. UCI Machine Learning Repository.
   https://doi.org/10.24432/C51W2X (CC BY 4.0).
3. Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T. *J. Korean Statistical Society*, 42, 323–328.
4. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830 (reference model,
   metrics, splitting).

## Appendix — complexity

Notation: n rows, d inputs, I iterations.

| Step | Time | Memory |
|---|---|---|
| Training (batch gradient descent) | O(I·n·d) — each iteration is one pass of X·w and Xᵀ·r. Here I ≈ 27k, n = 4,004, d = 38 | O(n·d) data + O(d) model |
| Prediction | O(d) per row | O(d) |
| Mini-batch variant | O(epochs · n · d) | O(batch · d) working memory |

Worst case the number of iterations I is set by the 30,000 cap; it is large here because the problem is badly
conditioned (some levels nearly separate the classes, pushing the optimum far from zero).
