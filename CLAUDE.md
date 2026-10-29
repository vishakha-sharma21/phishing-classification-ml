# CLAUDE.md — Phishing Website Detection Using Custom ML Algorithms

## 0. Project facts established in Phase 1 (verified from data)

- Local dataset: `data/raw/phishing.csv` (11,055 × 31, SHA-256 `35dd9aa6…59c2`), identical in value to the
  official `Training Dataset.arff`. Column names are lowercase snake_case; the target column is **`result`**.
- `fetch_ucirepo(id=327)` fails locally with an SSL certificate error in Python's trust store; the same
  `data_url` was downloaded with `curl` (verification on). See `reports/dataset_inspection.md`.
- 5,206 exact duplicate rows; 64 conflicting feature vectors. Proposed policy and status: `reports/cleaning_log.json`.
- `redirect` is coded {0,1}; `domain_registration_length` correlates opposite to its documented rule.
- Official feature description: `docs/source/`.
- APPROVED (after Phase 1): duplicate policy (collapse exact dups, drop conflicting vectors, then split) and
  label mapping -1 → phishing (1), +1 → legitimate (0). Cleaned data: 5,721 rows; split 4,004 / 858 / 859.
- Phase 2 (EDA, `src/eda.py`, `notebooks/eda.ipynb`): label-dependent stats use train only; test untouched.
  `prefix_suffix = 1` is 100% legitimate (all phishing rows are -1); `sslfinal_state`, `having_sub_domain`,
  `web_traffic` are non-monotone in their codes (motivates one-hot for LR).
- Phases 3–11 completed on user instruction ("finish the project"). Pipeline: `python -m src.train`
  (stages prepare, baselines, lr, tree, rf, evaluate, overfit, interpret) → `reports/results.json`.
  Selected: LR one-hot λ=1e-4 lr=1/L thr 0.35; RF 100 trees, depth ∞, leaf 1, max_features 15, thr 0.46.
  Docs (`README.md`, `reports/report.md`, `reports/viva_qa.md`) quote numbers from results.json — re-check them
  with `python -m src.check_docs` after any retraining.
- App update (user instruction, 2026-10-28): the app uses separate 8-input "quick" models (`--stage quick`;
  top 8 by training MI; `models/quick_*.json`), a three-zone verdict (cut-offs on validation, 98% recall/precision
  targets), an LR contribution panel (w·(x − train mean)) and an agreement badge. The user asked to REMOVE all
  limitations/disclaimer text from the app — this overrides Phase 10's "display limitations" rule; limitations
  live in `reports/report.md` §14 only.
- LR-only presentation (user instruction, 2026-10-28): the Decision Tree and Random Forest are REMOVED from every
  deliverable (app, report.md, viva_qa.md, README, Word report, PPT, full-pipeline notebook). Logistic Regression is
  the only model presented; this overrides §1's "two primary algorithms". The app uses the quick 8-input LR only
  (gauge = LR, cut-offs 0.17 / 0.92, no agreement badge). `src/decision_tree.py`, `src/random_forest.py`, their
  tests and the `tree` / `rf` / RF parts of `src.train` still exist and run, but are not reported.
  LR-only figures: `python -m src.lr_only_figures` (lr_05–lr_10).

## 1. Project Overview

**Project title:** Phishing Website Detection Using From-Scratch Machine Learning Algorithms

**Academic context:** 7th-semester Computer Engineering ML Project-Based Learning (PBL).

### Objective

Develop a complete machine learning system that classifies websites as:

- **Phishing:** A website exhibiting characteristics associated with phishing.
- **Legitimate:** A website exhibiting characteristics associated with legitimate websites.

Use two primary algorithms implemented independently from scratch in NumPy:

1. **Logistic Regression:** Binary classification using gradient descent, binary cross-entropy, and optional L2 regularization.
2. **Random Forest:** An ensemble of custom Gini-based Decision Trees using bootstrap sampling and random feature selection.

A custom Decision Tree is a required supporting component of the Random Forest, not a third primary model.

The project must demonstrate understanding of algorithm mechanics, data preparation, feature engineering, experimental design, evaluation, overfitting, and model interpretation.

### Required deliverables

- Reproducible dataset inspection and EDA.
- Clean preprocessing and a leakage-safe evaluation pipeline.
- Logistic Regression implemented from scratch.
- Decision Tree and Random Forest implemented from scratch.
- Comparison with scikit-learn reference implementations.
- Hyperparameter tuning and threshold optimization.
- Robust evaluation with appropriate metrics and confidence intervals.
- Feature selection and ablation studies.
- A functional Streamlit application.
- A technical report, README, and viva preparation notes.
- Unit tests and verification of the custom implementations.

**Important:** Do not optimize for a predetermined accuracy. Optimize for a correct, transparent, reproducible experiment and report the actual results, including negative findings.

---

## 2. Dataset

### Primary dataset: UCI Phishing Websites

Source: https://archive.ics.uci.edu/dataset/327/phishing+websites

Original research: Mohammad, R., McCluskey, L., and Thabtah, F. (2012), *An Assessment of Features Related to Phishing Websites Using an Automated Technique*.

DOI: https://doi.org/10.24432/C51W2X

License: Creative Commons Attribution 4.0 International (CC BY 4.0). Include appropriate attribution.

### Expected dataset properties — verify from the downloaded data

- Approximately 11,055 observations.
- 30 predictor features.
- One target column, commonly named `Result`.
- Feature values are generally encoded using {-1, 0, 1}.
- The original target convention is -1 for phishing and +1 for legitimate.
- No missing values are reported by UCI.

These are expected properties, not hardcoded assumptions. Inspect the actual downloaded file and document every discrepancy.

### Dataset acquisition

Preferred method:

```bash
pip install ucimlrepo
```

```python
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=327)

X = dataset.data.features
y = dataset.data.targets
```

Save a local, reproducible copy to:

`data/raw/phishing.csv`

Alternatively, download the original ARFF file from UCI and convert it to CSV.

Once downloaded, subsequent experiments must run offline using the saved dataset.

Never fabricate a dataset or silently substitute a different phishing dataset.

### Target mapping

Use a consistent convention throughout the project:

- `1 = phishing` — positive class.
- `0 = legitimate` — negative class.

If the original labels are -1 and +1, map them explicitly:

- Original -1 → phishing → 1.
- Original +1 → legitimate → 0.

Validate the mapping against the source documentation and observed labels. Save the mapping in the preprocessing configuration and report.

### Feature documentation

Use the feature description document linked from UCI to create:

`reports/features.md`

For each feature, document:

- Original feature name.
- Plain-English meaning.
- Permitted values.
- Whether it is a URL, domain, HTML/JavaScript, or other website indicator.
- Its limitations and any ambiguity in interpretation.

Do not invent feature meanings or claim that an association is causal.

---

## 3. Critical Data Quality and Leakage Requirements

The classic UCI phishing dataset can contain duplicate observations. Identical rows appearing in both training and test sets can make evaluation look artificially strong.

Investigate this before fitting any model.

### Duplicate audit

During Phase 1, calculate:

1. Number of original rows.
2. Exact duplicate rows across features and label.
3. Unique feature vectors.
4. Feature-identical groups with conflicting labels.
5. Number of rows remaining under the chosen duplicate-handling policy.
6. Class counts before and after cleaning.

Do not assume that every duplicate is an error. Document whether duplicates could represent repeated observations, redundant records, or ambiguous labels.

Use a defensible policy:

- Collapse exact feature-and-label duplicates when treating the dataset as a collection of unique feature vectors.
- Investigate feature-identical rows with conflicting labels separately.
- Do not arbitrarily choose a label for a conflicting group.
- If conflicting groups are excluded, record the number excluded and the rationale.
- Save an audit trail in `reports/cleaning_log.json`.

Deduplicate before splitting for the primary unique-record evaluation. Also explain how this changes the population being evaluated.

### Leakage investigation

Perform the following checks:

- Search for columns that duplicate or directly encode the target.
- Check feature-label associations for suspiciously strong predictors.
- Train single-feature diagnostic models using validation data.
- Investigate any feature with unusually high standalone performance.
- Compare results before and after duplicate handling as a documented diagnostic.
- Confirm that feature selection, encoding, scaling, resampling, and threshold tuning never use test labels.

A single-feature ROC-AUC close to 1.0 is a warning to investigate, not automatic proof of leakage. Some phishing indicators can genuinely be highly predictive.

Never delete a feature solely because it produces a strong score. Establish the reason and document the decision.

### Expected performance

Do not hardcode an expected accuracy, precision, recall, F1-score, or ROC-AUC.

Published results are not directly comparable unless their dataset cleaning, splits, feature sets, and evaluation protocols match.

If scores approach 99–100%, investigate duplicates, target leakage, accidental test-set reuse, and inconsistent preprocessing before accepting them.

---

## 4. Technical Rules

### Custom algorithms

Implement the core algorithms in NumPy:

- Logistic Regression.
- Gini-based Decision Tree.
- Random Forest.

Do not call scikit-learn's model `.fit()` inside any custom implementation.

Allowed uses of scikit-learn:

- Train/validation/test splitting.
- Metrics and diagnostic utilities.
- Optional stratified fold generation, provided the custom models perform the actual fitting.
- A separate reference-model comparison script.
- Independent validation of custom predictions.

Do not disguise scikit-learn model predictions as custom predictions.

### Coding standards

- Use clear, modular Python code.
- Add type hints and docstrings to public methods.
- Use a fixed random seed where appropriate.
- Avoid global mutable state.
- Separate training, prediction, evaluation, and visualization.
- Save fitted preprocessing parameters with model artifacts.
- Keep feature ordering consistent between training and inference.
- Handle numerical instability and invalid input explicitly.
- Make runtime and memory usage measurable.

Reuse correct implementations from an existing project if available. Inspect and test them before adapting them.

---

## 5. Project Structure

```text
phishing-detection/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── phishing.csv
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── cross_validation.py
│   ├── feature_selection.py
│   ├── threshold_tuning.py
│   ├── evaluation.py
│   └── train.py
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_logistic_regression.py
│   ├── test_decision_tree.py
│   ├── test_random_forest.py
│   └── test_evaluation.py
├── models/
│   ├── logistic_regression.json
│   ├── random_forest.json
│   └── preprocessing.json
├── reports/
│   ├── figures/
│   ├── dataset_inspection.md
│   ├── cleaning_log.json
│   ├── features.md
│   ├── results.json
│   ├── report.md
│   └── viva_qa.md
└── app.py
```

Adjust the structure only when there is a concrete reason. Keep the project simple enough to maintain and explain.

Deviations so far: `src/inspect_dataset.py` (Phase 1 script) and `docs/source/` (official UCI feature document and metadata).

---

## 6. Phase 1 — Dataset Inspection

Do not train models in this phase.

Tasks:

1. Inspect the repository and existing files before making changes.
2. Load the original dataset.
3. Verify the row count, column count, column names, and target column.
4. Inspect data types and unique values for each feature.
5. Verify that predictor values match the documented encoding.
6. Inspect missing values and invalid values.
7. Calculate the original class distribution and class percentages.
8. Audit exact duplicates and feature-identical conflicting labels.
9. Record the chosen cleaning policy and estimated retained dataset size.
10. Check for unexpected target-like columns or suspicious features.
11. Verify that the source and license are documented.

Create:

- `reports/dataset_inspection.md`
- `reports/cleaning_log.json`

Include actual values from the executed code.

At the end of this phase, run the inspection script, show its output, explain the findings, and stop for approval.

---

## 7. Phase 2 — Exploratory Data Analysis

Generate reproducible plots and save all figures to `reports/figures/`.

Required analysis:

### Dataset overview

- Class distribution and percentages.
- Dataset dimensions before and after cleaning.
- Duplicate statistics.
- Per-feature value counts.

### Feature analysis

- Stacked bar plots showing feature values by target class.
- Phishing rate for each value of each categorical feature.
- Feature-target association summaries.
- Correlation or association heatmap, using an appropriate measure for discrete variables.
- A ranking of single-feature validation ROC-AUC.
- Feature distributions grouped by phishing and legitimate labels.

### Interpretation

For every figure:

- Explain what it shows.
- Identify the visible pattern.
- State any limitations.
- Avoid causal conclusions.
- Clearly distinguish exploratory findings from validated predictive performance.

Use validation-only model-based diagnostics where relevant. Do not select features by inspecting test-set performance.

Create:

`notebooks/eda.ipynb`

and save reusable plots under:

`reports/figures/`

Finish by running the notebook or its equivalent scripts and summarize the actual findings.

Stop for approval.

---

## 8. Phase 3 — Preprocessing, Encoding and Splitting

### Cleaning

- Apply the approved duplicate policy.
- Resolve or exclude conflicting-label groups according to the documented policy.
- Map labels to phishing = 1 and legitimate = 0.
- Validate all feature columns and target labels.
- Preserve the original data and write cleaned data separately.

### Feature encoding

The predictors use discrete values, generally -1, 0, and 1.

For Logistic Regression:

- Compare raw numerical coding against one-hot encoding for three-valued features.
- Justify the representation used in the final model.
- Handle the reference category consistently when one-hot encoding.
- Ensure feature names and ordering are saved.

For Random Forest:

- Use the original discrete codes by default.
- Explain how numerical split thresholds divide the encoded values.
- Optionally compare raw codes with an explicitly categorical split strategy if time permits.

Scaling is not inherently required for these bounded discrete features. If scaling or another transformation is used, justify it and fit it on training data only.

### Splitting

Use:

- 70% training.
- 15% validation.
- 15% final test.
- Random seed = 42.
- Stratification by target class.

Perform the split after applying the approved deduplication policy.

If a custom splitting strategy is introduced, explain why it is preferable and preserve the untouched final test set.

Save split indices or sufficient metadata to reproduce the split.

### Baselines

Implement and report:

1. Majority-class baseline.
2. Best single-feature baseline.
3. Logistic Regression reference baseline, clearly marked as a reference model rather than the custom implementation.

Report accuracy, precision, recall, F1-score, ROC-AUC and PR-AUC where defined.

Explain why accuracy alone is insufficient.

Create a preprocessing summary and stop for approval.

---

## 9. Phase 4 — Logistic Regression From Scratch

Implement binary Logistic Regression using NumPy.

Required components:

1. Parameter initialization.
2. Stable sigmoid function.
3. Linear logit calculation.
4. Binary cross-entropy loss with numerical safeguards.
5. Gradient computation for weights and bias.
6. Batch gradient descent.
7. Optional mini-batch gradient descent.
8. Learning-rate configuration.
9. Optional L2 regularization.
10. Optional class-weighted loss.
11. `fit(X, y)`.
12. `predict_proba(X)`.
13. `predict(X, threshold=...)`.
14. Serialization and reload.

### Mathematical requirements

Document the sigmoid, binary cross-entropy, gradient expressions, and regularization term.

Ensure the regularization term is applied consistently to the loss and gradient, excluding the bias unless otherwise justified.

Use numerically stable calculations to avoid overflow and invalid logarithms.

### Testing

- Verify output dimensions.
- Verify probabilities remain finite and within [0, 1].
- Check convergence on simple synthetic datasets.
- Implement a numerical gradient-check test.
- Test prediction thresholds.
- Test save/load equivalence.
- Compare custom results against a separately trained scikit-learn reference model.

### Experiments

Investigate:

- Learning rate.
- Number of iterations.
- L2 regularization strength.
- Raw feature coding versus one-hot encoding.
- Optional class weighting.
- Validation threshold.

Save training and validation loss curves.

Do not choose hyperparameters based on test performance.

Create `src/logistic_regression.py`, its tests, and the relevant figures.

Stop for approval.

---

## 10. Phase 5 — Decision Tree From Scratch

Build a reusable Decision Tree as the foundation for Random Forest.

Implement:

- Gini impurity.
- Weighted impurity after a split.
- Best-split search over candidate feature thresholds.
- Recursive tree construction.
- Maximum depth.
- Minimum samples required to split.
- Minimum samples per leaf.
- Optional maximum number of candidate features.
- Stopping conditions.
- Leaf class probabilities.
- `predict_proba()` and `predict()`.
- Feature importance.
- JSON serialization and reload.

### Correctness requirements

- Handle pure leaves and empty candidate splits safely.
- Avoid infinite recursion.
- Use deterministic tie-breaking where possible.
- Check input dimensions and feature ordering.
- Do not silently accept malformed input.
- Clearly document how thresholds are selected.

### Tests

Use small synthetic datasets with known patterns.

Test:

- Perfectly separable examples.
- Pure nodes.
- Constant features.
- Minimum leaf constraints.
- Maximum depth.
- Probability output.
- Serialization and reload.
- Comparison with scikit-learn on small datasets using compatible settings.

Document any unavoidable implementation differences.

Create `src/decision_tree.py` and the associated tests.

Stop for approval.

---

## 11. Phase 6 — Random Forest From Scratch

Implement Random Forest using the custom Decision Tree.

### Required functionality

1. Generate bootstrap samples from the training data.
2. Train an independently randomized Decision Tree for each tree.
3. Select random subsets of features during split search.
4. Support configurable tree count.
5. Support configurable maximum depth and minimum leaf size.
6. Aggregate tree predictions into class probabilities.
7. Apply a configurable threshold to the aggregated phishing probability.
8. Calculate feature importance.
9. Save and reload the complete forest.
10. Record training time and inference time.

### Hyperparameter experiments

Study:

- Number of trees.
- Maximum depth.
- Minimum samples per leaf.
- Number of candidate features per split.
- Bootstrap versus no-bootstrap diagnostic, if computationally feasible.

Use validation data to select configurations.

Explain why Random Forest reduces the variance of an individual Decision Tree and why random feature selection encourages tree diversity.

Compare:

- Custom Random Forest.
- Custom Logistic Regression.
- A separate scikit-learn Random Forest reference.
- A majority-class baseline.

Do not use a reference model's predictions inside the custom forest.

### Efficiency

Measure training time and memory use.

Avoid needlessly expensive repeated computations in split search. If the custom implementation is too slow, optimize it transparently and test that the changes preserve correctness.

Do not silently switch to scikit-learn model fitting.

Create `src/random_forest.py` and the relevant tests.

Stop for approval.

---

## 12. Phase 7 — Evaluation and Model Comparison

Finalize model selection using validation results before evaluating the test set.

Evaluate the selected configurations on the untouched test set.

### Required metrics

Report:

- Accuracy.
- Precision for the phishing class.
- Recall for the phishing class.
- F1-score for the phishing class.
- Macro-F1.
- ROC-AUC.
- PR-AUC / average precision.
- Confusion matrix.
- True positive, false positive, true negative and false negative counts.

Specify which class is treated as positive.

Use the saved phishing probability outputs to calculate ranking metrics. Do not calculate ROC-AUC from hard class predictions when probability scores are available.

### Visualizations

Save:

- ROC curves.
- Precision-recall curves.
- Confusion-matrix heatmaps.
- Model comparison chart.
- Per-class metric chart.
- Threshold-versus-precision/recall/F1 chart.
- Training-versus-validation learning curves where applicable.

### Statistical reliability

Where computationally feasible:

- Estimate 95% bootstrap confidence intervals for key test metrics.
- Use stratified five-fold cross-validation on the training partition for model comparison.
- Keep the validation and final test partitions out of training-fold fitting.
- Document random seeds and sampling methodology.

Do not present overlapping confidence intervals as a definitive significance test.

### Interpretation

Write a short, evidence-based verdict for each comparison:

- Clear improvement.
- Small or uncertain improvement.
- No meaningful improvement.
- Worse performance.

Distinguish threshold-dependent metrics from ranking metrics.

A model with a higher ROC-AUC does not necessarily have better precision or recall at the selected threshold.

Create `reports/results.json` programmatically from actual experiment outputs.

Stop for approval.

---

## 13. Phase 8 — Overfitting and Underfitting

Investigate whether the models generalize beyond the training set.

### Logistic Regression

Study:

- Training and validation loss.
- Learning rate.
- Iteration count.
- L2 regularization strength.
- Raw versus one-hot feature coding.
- Performance at different decision thresholds.

### Decision Tree and Random Forest

Study:

- Training versus validation metrics at different depths.
- Minimum samples per leaf.
- Number of trees.
- Feature subset size.
- Difference between an unconstrained tree and a tuned tree.

### Feature-selection experiments

Implement at least two transparent feature-selection approaches:

- Training-set variance filtering.
- Training-only feature ranking using an appropriate statistical association measure or validation-based single-feature score.

Do not use test labels to select features.

Compare:

- All features.
- Reduced feature set.
- Performance change.
- Runtime change.

Feature selection should be label-free when specified, or use training data only when the method is supervised.

Explain when feature reduction helps, when it harms performance, and whether it reduces complexity.

Stop for approval.

---

## 14. Phase 9 — Model Interpretation

### Logistic Regression

- Display coefficients with feature names.
- Explain the sign of each coefficient.
- Explain how regularization affects coefficient magnitude.
- If one-hot encoding is used, interpret coefficients relative to the reference category.
- Avoid treating coefficients as causal effects.

### Random Forest

- Calculate impurity-based feature importance.
- Compare the most important features across the forest and Logistic Regression.
- Where useful, calculate permutation importance using validation data.
- Investigate whether feature importance is unstable across different seeds or folds.

Use the UCI feature descriptions to explain the practical meaning of important predictors.

Distinguish predictive association from causation.

Save plots and summaries under `reports/figures/` and `reports/`.

Stop for approval.

---

## 15. Phase 10 — Streamlit Application

Build an educational demonstration application in `app.py`.

### Requirements

- Load saved models without retraining on application startup.
- Load the exact preprocessing configuration used during training.
- Preserve feature order.
- Validate all inputs and reject missing or invalid values.
- Support manual feature entry through labeled dropdowns.
- Support CSV upload for batch prediction.
- Display Logistic Regression and Random Forest predictions side by side.
- Display phishing probability and the selected operating threshold.
- Display the predicted class.
- Show saved evaluation metrics and confusion matrices.
- Display feature importance and model limitations.
- Explain that outputs represent model-estimated phishing risk, not a guarantee.

### Feature input

Use plain-English descriptions wherever supported by the official feature documentation.

Do not pretend that the original dataset's 30 engineered features are the same as features extracted from an arbitrary live URL.

If users enter a URL, do not feed the raw URL directly into the trained models. Either:

1. Explain that the current version accepts pre-extracted features only; or
2. Implement a separate, documented feature extraction pipeline that generates compatible features and validate it independently.

Do not silently map unavailable features to fabricated values.

Do not automatically visit suspicious URLs or execute their content.

Include a clear limitation that the dataset is old and that its predictions do not establish the safety of a currently active website.

Create and test the application before moving to documentation.

Stop for approval.

---

## 16. Phase 11 — Documentation and Viva Preparation

### README.md

Include:

- Problem statement.
- Dataset source and citation.
- Setup instructions.
- Project structure.
- Training commands.
- Testing commands.
- Streamlit launch instructions.
- Actual measured results.
- Known limitations.

### reports/report.md

Include:

1. Abstract.
2. Introduction and motivation.
3. Problem definition.
4. Dataset source and license.
5. Feature descriptions.
6. Dataset inspection and duplicate findings.
7. Preprocessing and split strategy.
8. Algorithm design.
9. From-scratch implementation details.
10. Experimental methodology.
11. Evaluation results.
12. Overfitting analysis.
13. Feature interpretation.
14. Limitations.
15. Future scope.
16. Conclusion and references.

### reports/viva_qa.md

Prepare beginner-friendly questions and answers on:

- What phishing is and why detection matters.
- Dataset source, features and labels.
- Why target labels are remapped.
- Sigmoid function.
- Binary cross-entropy.
- Gradient descent.
- Derivation of Logistic Regression gradients.
- L2 regularization.
- Why raw ternary encoding may be questionable for a linear model.
- Why scaling may be unnecessary for the tree model.
- Gini impurity.
- How a tree selects a split.
- Bootstrap sampling.
- Random feature selection.
- Why a forest can outperform a single tree.
- Precision, recall, F1-score, ROC-AUC and PR-AUC.
- Threshold selection.
- Data leakage and duplicate handling.
- Train/validation/test separation.
- Cross-validation.
- Overfitting and underfitting.
- Feature importance and its limitations.
- Custom implementation verification.
- Time and space complexity.
- Dataset age and limitations.
- Why the application is not a guarantee of website safety.

### Complexity analysis

Document approximate time and space complexity for:

- Logistic Regression training and inference.
- Decision Tree split search and inference.
- Random Forest training and inference.

State the assumptions behind each complexity estimate. Distinguish worst-case behavior from typical behavior.

Ensure every reported number matches the generated results files.

---

## 17. Requirements and Commands

Maintain a minimal `requirements.txt` containing compatible versions of the required libraries, such as:

- numpy
- pandas
- matplotlib
- seaborn
- streamlit
- pytest
- ucimlrepo
- scipy

Scikit-learn may be included for evaluation utilities and reference-model comparisons.

Do not add large dependencies without justification.

### Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Tests

```bash
pytest -q
```

### Train and evaluate

```bash
python -m src.train
```

### Launch app

```bash
streamlit run app.py
```

Training must regenerate saved models, `reports/results.json`, and the required evaluation figures.

The application must work with saved models and must not require retraining at startup.

---

## 18. Working Style and Approval Gates

Work strictly one phase at a time.

Before each phase:

1. Inspect the current repository and existing implementations.
2. State the objective and files expected to change.
3. Identify dependencies on previous phases.

At the end of each phase:

1. Execute the relevant code.
2. Run relevant tests.
3. Show actual terminal output and generated files.
4. Summarize the work in simple language.
5. Explain what was verified and what remains unresolved.
6. Stop and wait for my explicit approval.

Do not begin the next phase until I approve.

Do not claim that tests passed or results were generated unless the commands were actually run.

If a command fails, explain the error and fix it within the current phase where possible.

Commit to Git after each approved phase if a Git repository is initialized. Never commit the dataset, virtual environment, credentials, or other sensitive files.

Do not overwrite existing work without inspecting it first.

---

## 19. Definition of Done

The project is complete only when:

- The dataset is verified and its provenance is documented.
- Duplicate handling and leakage checks are documented.
- Logistic Regression is implemented independently in NumPy.
- The custom Decision Tree and Random Forest work independently of scikit-learn model fitting.
- Unit tests and numerical gradient checks pass.
- The custom models are compared fairly against reference implementations.
- Model selection uses validation data only.
- Final test evaluation uses an untouched test set.
- All required metrics and figures are generated from actual predictions.
- Overfitting and feature-selection experiments are documented.
- The Streamlit app loads saved models and validates input.
- The README, technical report and viva notes reflect the actual implementation.
- All reported numbers are traceable to executed experiments.
- Limitations and unsuccessful experiments are reported honestly.
