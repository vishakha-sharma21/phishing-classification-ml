# Phishing Website Detection — From-Scratch Logistic Regression

Classify websites as **phishing** or **legitimate** from the 30 engineered indicators of the UCI *Phishing Websites*
dataset with **Logistic Regression implemented from scratch in NumPy** (gradient descent, binary cross-entropy, L2).
A leakage-safe pipeline, honest evaluation and a Streamlit demo complete it. (7th-semester Computer Engineering ML PBL.)

## Dataset and citation

Mohammad, R., McCluskey, L. (2012). *Phishing Websites* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C51W2X — licensed **CC BY 4.0**.
Intro paper: Mohammad, Thabtah & McCluskey (2012), *An assessment of features related to phishing websites using an
automated technique*, ICITST.

The dataset is **not committed**. Download it once, after which everything runs offline:

```bash
python -m src.data_loading      # uses ucimlrepo; saves data/raw/phishing.csv
```

If `ucimlrepo` fails with an SSL certificate error (it did on the development machine), download the identical file
directly and save it as `data/raw/phishing.csv`:

```bash
curl -o data/raw/phishing.csv https://archive.ics.uci.edu/static/public/327/data.csv
```

Expected SHA-256: `35dd9aa6870643c292d68343d926869095292a7406d198305cae7435069b59c2`.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Commands

| Step | Command | Output |
|---|---|---|
| Dataset inspection (Phase 1) | `python -m src.inspect_dataset` | `reports/dataset_inspection.md`, `reports/cleaning_log.json` |
| EDA (Phase 2) | `python -m src.eda` (or run `notebooks/eda.ipynb`) | `reports/figures/0*.png`, `reports/eda_results.json` |
| Feature docs | `python -m src.feature_docs` | `reports/features.md` |
| Whole project in one notebook | open `notebooks/phishing_detection_full_pipeline.ipynb` | data → EDA → model → evaluation → overfitting → app model → tests |
| Train + evaluate everything | `python -m src.train` (~25 min) | `models/*.json`, `reports/results.json`, figures |
| One stage only | `python -m src.train --stage lr` | stages: `prepare baselines lr tree rf evaluate overfit interpret quick` |
| Logistic-Regression-only figures | `python -m src.lr_only_figures` | `reports/figures/lr_05`–`lr_10` |
| Tests | `pytest -q` | 71 tests |
| App | `streamlit run app.py` | loads the saved 8-input Logistic Regression; no retraining |

## Project structure

```text
src/
  config.py               paths, seed, label mapping (-1 → phishing=1, +1 → legitimate=0)
  data_loading.py         download / load raw CSV
  inspect_dataset.py      Phase 1 audit (duplicates, conflicts, leakage screen)
  eda.py                  Phase 2 figures and statistics (train-only label stats)
  preprocessing.py        cleaning, stratified 70/15/15 split, raw / one-hot encoders, input validation
  logistic_regression.py  custom LR (stable sigmoid, BCE, L2, class weights, mini-batch, JSON)
  decision_tree.py,       an earlier tree / forest comparison; still run by the pipeline but not part of the
  random_forest.py        reported results
  evaluation.py           metrics, bootstrap CIs, paired bootstrap differences, verdicts
  threshold_tuning.py     validation threshold selection and curves
  cross_validation.py     stratified 5-fold CV on the training partition
  feature_selection.py    near-constant filter, mutual-information ranking (train only)
  interpretation.py       permutation importance, rank stability
  reference_models.py     baselines + scikit-learn REFERENCE models (never used inside custom models)
  plotting.py             figure style and evaluation plots
  feature_info.py         plain-English feature metadata from the UCI document
  feature_docs.py         generates reports/features.md
  inference.py            load the saved app model, validate and score inputs, LR contributions
  lr_only_figures.py      Logistic-Regression-only evaluation, overfitting and importance figures
  train.py                end-to-end pipeline (stages)
tests/                    unit tests for every module + Streamlit AppTest
models/                   preprocessing.json, logistic_regression.json, quick_logistic_regression.json
reports/                  inspection, features, results.json, report.md, viva_qa.md, figures/
notebooks/                eda.ipynb, phishing_detection_full_pipeline.ipynb
docs/source/              official UCI feature document and metadata
app.py                    Streamlit app
```

## Results (untouched test set, 859 unique rows; positive = phishing)

The model is evaluated once at its validation-chosen threshold. Brackets give 95% bootstrap intervals.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Custom Logistic Regression** | 0.934 | 0.918 | 0.957 | **0.937 [0.920, 0.952]** | **0.988 [0.983, 0.992]** | 0.989 |
| scikit-learn LR (reference) | 0.934 | 0.918 | 0.957 | 0.937 | 0.988 | 0.989 |
| Best single feature (`sslfinal_state`) | 0.885 | 0.897 | 0.878 | 0.887 | 0.897 | 0.874 |
| Majority class | 0.517 | 0.517 | 1.000 | 0.681 | 0.500 | 0.517 |

- **vs the best single feature:** the paired bootstrap gives ΔF1 = +0.050 [+0.029, +0.070], a clear improvement.
- **Custom vs scikit-learn:** no meaningful difference (identical test F1).
- **Where the numbers come from:** full details in `reports/results.json` and `reports/report.md`.

## The app (8 questions)

The Streamlit app asks only **8 questions** — the top 8 features by training-set mutual information — and scores them
with a separate 8-input Logistic Regression (`models/quick_logistic_regression.json`, built by
`python -m src.train --stage quick`). It shows:

- a colour **risk gauge** with the phishing probability;
- a **three-level verdict** — Safe (< 17%), Suspicious, Likely phishing (≥ 92%) — with cut-offs chosen on validation;
- a **"Why this result?"** chart of each answer's push on the score, and the main reason in plain English.

| Model (test set) | Inputs | F1 [95% CI] | ROC-AUC [95% CI] |
|---|---|---|---|
| Logistic Regression (app) | 8 | 0.932 [0.914, 0.948] | 0.987 [0.982, 0.991] |
| Logistic Regression | 30 | 0.937 [0.920, 0.952] | 0.988 [0.983, 0.992] |

8 vs 30 inputs: ΔF1 −0.005 [−0.019, +0.006] — no clear loss.

## Key findings

- **Duplicates:** 47% of raw rows are exact duplicates and 64 feature vectors carry both labels. Cleaning leaves 5,721
  unique rows, and cleaning happens before splitting. Retraining on raw data did *not* inflate scores
  (validation accuracy 0.930 vs 0.951). This is reported as a negative finding.
- **Two strong features:** HTTPS certificate state and anchor-link origin dominate. Removing both costs 0.071 AUC.
- **Encoding:** one-hot beats raw −1/0/1 codes (validation log-loss 0.139 vs 0.192), because several features
  are non-monotone.

## Known limitations

- The data are from around 2012. Several features depend on retired services (Alexa rank, public PageRank), so
  predictions say nothing reliable about current websites.
- The models score **pre-extracted features only**. The app does not accept or visit URLs.
- A linear model cannot learn interactions between features unless they are engineered by hand.
- Some codings are undocumented or inconsistent with their description: `redirect`, `prefix_suffix` and
  `domain_registration_length` (see `reports/features.md`).
- Evaluation uses unique feature vectors, not real traffic. The probabilities are not calibrated to how common
  phishing actually is.
- Outputs are model-estimated risk, **not a guarantee** of safety.
