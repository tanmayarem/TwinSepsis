# Existing Experiments — TwinSepsis (PhysioNet 2019)

| Field | Value |
|---|---|
| Audit date | 2026-10-29 |
| Repository / commit | `C:\TwinSepsis` / `main` @ `a7659b6` |
| Scope | All model, clinical-score, and ensemble experiments currently recorded in the repository |
| Data used by every experiment below | PhysioNet Challenge 2019 only (40,336 patients, 1,552,210 hourly records) |
| MIMIC-IV | **Not used by any experiment.** There is no MIMIC code or result (see `repository_audit.md`, F-02). |
| Mode | Read-only. No notebook was executed and no result was regenerated, except the explicit metric checks noted as **[V] recomputed**. |
| Confidentiality | No patient identifiers, raw rows, or record-level extracts are reproduced here. |

Legend: **[V]** = verified from code, stored notebook output, or a result file, or independently recomputed; **[S]** = suspected/inferred, with the reason stated. Threshold-dependent metrics are always shown with the threshold that produced them.

---

## 1. How to read this document

This is an experiment-by-experiment record of **what was run**, **how it was selected**, **what was saved**, and **what can legitimately be cited**. It is not a re-analysis and it does not "fix" any result.

Two rules apply throughout:

1. **A result CSV is authoritative over a notebook output** when both exist. Notebook outputs are treated as evidence of what ran, not as a guarantee that the current code still produces the same number.
2. **A metric is only comparable to another metric at the same threshold and on the same unit of analysis.** All metrics here are hourly-row metrics unless explicitly labelled otherwise.

The companion documents are:

- `repository_audit.md` — repository-wide findings register (F-01 … F-24).
- `mimic_table_inventory.csv` — MIMIC-IV schema inventory (unused by these experiments).
- `preprocessing_plan.md` — proposed, separate MIMIC-IV pipeline. Nothing in it has been implemented.

---

## 2. Common data and split contract

### 2.1 Cohort used by every experiment

| Item | Value | Status |
|---|---|---|
| Patients | 40,336 (setA 20,336 + setB 20,000) | **[V]** |
| Hourly records | 1,552,210 | **[V]** |
| Positive hours | 27,916 (1.798%) | **[V]** |
| Patients with ≥1 positive hour | 2,932 (7.269%) | **[V]** |
| Feature columns | 40 (8 vitals, 26 labs, 6 demographic/admin) + `SepsisLabel` | **[V]** |
| Label | PhysioNet `SepsisLabel` as distributed; the 6-hour pre-onset convention is **not documented anywhere in this repository** | **[V]** for absence; label semantics are external specification |

### 2.2 Split sizes and composition

| Split | Patients | Hours | Positive hours | Positive rate | Septic patients | Sepsis rate |
|---|---:|---:|---:|---:|---:|---:|
| Train | 28,235 | 1,086,436 | 19,560 | 1.800% | 2,052 | 7.27% |
| Validation | 6,050 | 231,472 | 4,162 | 1.798% | 440 | 7.27% |
| Test | 6,051 | 234,302 | 4,194 | 1.790% | 440 | 7.27% |

- Zero duplicate IDs, zero pairwise overlap, union = 40,336. **[V]**
- The identical sepsis rate to two decimals and the near-50/50 setA/setB mix strongly suggest a stratified random **patient-level** split. **[S]** The generator is missing: `03_preprocessing_and_split.ipynb` is an empty 129-byte file. **[V]**
- The same three ID files are used by XGBoost, Random Forest, GRU, LSTM, Transformer, SOFA, SIRS, and the ensemble. No experiment uses a different split. **[V]**

### 2.3 Environment caveat

`torch` is **not** in `requirements.txt` and **not** installed in `.venv`. The GRU, LSTM, and Transformer notebooks were executed in other environments (macOS / different Windows machines / different Python versions). See `repository_audit.md` §4 and F-03/F-04. **[V]**

---

## 3. Experiment cards

### E-01 — XGBoost baseline

| Field | Detail |
|---|---|
| Notebook | `notebooks/04_xgboost.ipynb` |
| Model file | `models/xgboost_baseline.json` (1.54 MB, tracked) |
| Result files | `results/xgboost_validation_metrics.csv`, `xgboost_test_metrics.csv`, `xgboost_test_confusion_matrix.csv` |
| Task | Binary hourly sepsis classification |
| Unit of analysis | Patient-hour row (rows from the same patient are not grouped) |

**Features and preprocessing [V]**

- 40 raw columns; `SepsisLabel` and `patient_id` dropped.
- No imputation, no forward fill, no scaling, no missingness indicators, no temporal aggregation.
- `inf`/`-inf` replaced by `NaN`; XGBoost handles `NaN` natively.
- Only explicitly temporal inputs are `ICULOS` and `HospAdmTime`; the model has no history window.

**Training configuration [V]**

| Parameter | Value |
|---|---|
| `n_estimators` | 500 |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `objective` | `binary:logistic` |
| `eval_metric` | `auc` |
| `scale_pos_weight` | 54.54 (train negative / positive) |
| `tree_method` | `hist` |
| `random_state` | 42 |
| Early stopping | 30 rounds on validation AUC |
| Best iteration | **196** |
| Best validation AUC | **0.7966686543** |

**Validation metrics at threshold 0.5 [V]**

| Metric | Value |
|---|---:|
| ROC-AUC | 0.7967 |
| Accuracy | 0.8157 |
| Precision | 0.0591 |
| Recall | 0.6204 |
| F1 | 0.1080 |
| Confusion matrix | TN 186,222 / FP 41,088 / FN 1,580 / TP 2,582 |

**Test metrics at threshold 0.5 [V]**

| Metric | Value |
|---|---:|
| ROC-AUC | 0.816563 |
| AUPRC | 0.121752 (recomputed from the ensemble prediction file) |
| Accuracy | 0.814432 |
| Precision | 0.061933 |
| Recall | 0.662136 |
| F1 | 0.113271 |
| Confusion matrix | TN 188,046 / FP 42,062 / FN 1,417 / TP 2,777 |

**Selection and caveats**

- Early stopping uses the validation split, so validation AUC is optimistically selected; the test AUC is still a legitimate held-out number because no test-guided choice was made. **[V]**
- Threshold 0.5 is arbitrary and not tuned; the high recall / very low precision reflects the 54.54 positive-class weight. **[V]**
- XGBoost is the strongest single model by test AUC and AUPRC except for the ensemble.
- Feature importance / SHAP is present in the notebook but no SHAP artifact is saved in `results/`. **[V]**

---

### E-02 — Random Forest baseline

| Field | Detail |
|---|---|
| Notebook | `notebooks/05_random_forest.ipynb` |
| Model file | `models/random_forest_baseline.joblib` (1.04 GB, **untracked**, git-ignored) |
| Result files | `results/random_forest_test_metrics.csv`, `random_forest_test_confusion_matrix.csv` |
| Task | Binary hourly sepsis classification |

**Features and preprocessing [V]**

- Same 40 raw features as XGBoost.
- No imputation, no scaling, no temporal features, no missingness indicators.
- `inf`/`-inf` → `NaN`; scikit-learn's Random Forest does not natively accept `NaN` in older versions, but this run completed under the installed stack, so the effective behaviour is version-dependent and not documented in the notebook. **[V] for code; [S] for exact internal handling** — do not assume portability without re-running.

**Training configuration [V]**

| Parameter | Value |
|---|---|
| `n_estimators` | 300 |
| `class_weight` | `balanced_subsample` |
| `max_features` | `sqrt` |
| `min_samples_leaf` | 2 |
| `max_depth` | unlimited (default) |
| `random_state` | 42 |
| `n_jobs` | -1 |

**Validation metrics at threshold 0.5 [V]**

| Metric | Value |
|---|---:|
| ROC-AUC | 0.7631 |
| Accuracy | 0.9803 |
| Precision | 0.0738 |
| Recall | 0.0084 |
| F1 | 0.0151 |
| Confusion matrix | TN 226,871 / FP 439 / FN 4,127 / TP 35 |

At threshold 0.5 the model is almost entirely negative. This is a threshold artefact, not a useful operating point.

**Threshold selection [V]**

- Validation F1 was maximised over thresholds `0.01 … 0.50` in steps of 0.01.
- Selected threshold: **0.17**.
- Validation at 0.17: precision 0.073606, recall 0.255887, F1 0.114326.

**Test metrics at threshold 0.17 [V]**

| Metric | Value |
|---|---:|
| ROC-AUC | 0.781508 |
| AUPRC | not saved (no test probability file) |
| Accuracy | 0.929868 |
| Precision | 0.090812 |
| Recall | 0.323796 |
| F1 | 0.141842 |
| Confusion matrix | TN 216,512 / FP 13,596 / FN 2,836 / TP 1,358 |

**Selection and caveats**

- The threshold is validation-selected, so the test metrics at 0.17 are held-out with respect to threshold choice. **[V]**
- The 1.04 GB model is **not in git**, so the exact fitted model cannot be recovered from a clean clone. **[V]**
- No validation metrics file is saved; only the notebook output records validation at 0.5 and 0.17. **[V]**

---

### E-03 — Partial SOFA (rule-based, validation only)

| Field | Detail |
|---|---|
| Notebook | `notebooks/06_sofa.ipynb` |
| Result files | **None.** Metrics exist only in notebook outputs/screenshots. |
| Task | Apply a partial SOFA-style organ-dysfunction rule, then classify `Partial_SOFA >= 2` |
| Split | Validation only (231,472 rows); no test evaluation |

**What was implemented [V]**

| Canonical SOFA component | Implemented? | Variables used |
|---|---|---|
| Coagulation | Yes | Platelets |
| Liver | Yes | Total bilirubin |
| Cardiovascular | **Partial** | MAP only; no vasopressor dose |
| Renal | **Partial** | Creatinine only; no urine output |
| Respiratory | **No** | PaO2/FiO2 absent |
| CNS | **No** | GCS absent |

- Missing components are left as `NaN` and skipped when summing; `SOFA_Components_Available` is recorded. Scores are therefore **not comparable across rows**. **[V]**
- Boundary errors against the canonical SOFA table: **[V]**
  - Bilirubin `<= 12.0` is scored 3; canonical `>= 12.0` is 4.
  - Creatinine `<= 5.0` is scored 3; canonical `>= 5.0` is 4.
- The notebook itself records "Canonical SOFA reproducible? — No". **[V]**

**Coverage [V]**

- Rows with at least one component: **204,102** of 231,472 (88.2%).
- Rows with all four components: **2,831** (1.2%).

**Results — `Partial_SOFA >= 2` [V]**

| Analysis | Rows | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Any available component | 204,102 | 179 | 5,194 | 3,661 | 195,068 | 0.0333 | 0.0466 | 0.0389 | 0.9566 |
| All four components | 2,831 | 74 | 1,449 | 31 | 1,277 | 0.0486 | 0.7048 | 0.0909 | 0.4772 |

Precision/recall/F1/accuracy in this table are **derived from the saved confusion matrices**; they were not written to a metric file. The notebook did not persist an AUC for the summed partial score.

**Why it must not be called SOFA**

- It is not the Sepsis-3 organ-dysfunction criterion and cannot be used to derive a Sepsis-3 label. **[S/V]** — the missing respiratory/CNS components and MAP-only/creatinine-only substitutions are verified; the consequence is a direct inference.
- The two boundary errors change scores at clinically relevant cut-points. **[V]**
- The rule predicts relatively few positives overall; on the complete-component subset it flags more than half the rows because the subset is small and enriched. The two rows of the table are not comparable populations. **[V]**

---

### E-04 — Modified SIRS (rule-based, validation only)

| Field | Detail |
|---|---|
| Notebook | `notebooks/07_sirs.ipynb` |
| Result files | **None.** Metrics exist only in notebook outputs. |
| Task | Apply four SIRS criteria, then classify `Modified_SIRS >= 2` |
| Split | Validation only (231,472 rows); no test evaluation |

**Criteria implemented [V]**

| Criterion | Rule |
|---|---|
| Temperature | `Temp > 38` or `Temp < 36` °C |
| Heart rate | `HR > 90` |
| Respiratory | `Resp > 20` or `PaCO2 < 32` |
| WBC | `WBC > 12` or `WBC < 4` (×10³/µL) |

- Band forms are unavailable. The score is therefore a **modified** SIRS, not the full clinical definition. **[V]**
- Missing criteria are skipped (`min_count=1`), so a row with only one available criterion can never reach 2 and is effectively treated as negative rather than unassessable. This makes the score availability-dependent. **[V]**

**Coverage [V]**

- Rows with at least one criterion: **211,124** of 231,472 (91.2%).
- Rows with all four criteria: **5,380** (2.3%).

**Results — `Modified_SIRS >= 2` [V]**

| Analysis | Rows | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Any available criterion | 211,124 | 1,163 | 32,777 | 2,783 | 174,401 | 0.0343 | 0.2947 | 0.0614 | 0.8316 |
| All four criteria | 5,380 | 59 | 1,912 | 55 | 3,354 | 0.0299 | 0.5175 | 0.0566 | 0.6344 |

Precision/recall/F1/accuracy are derived from the saved confusion matrices; no metric file was written.

**The "ROC_AUC" column is not an AUC [V]**

- For individual criteria, `y_score` is the binary 0/1 criterion value, not a continuous score. `roc_auc_score(y_true, y_score)` on a binary score is arithmetically `(sensitivity + specificity) / 2`.
- Any table that reports these values as "ROC-AUC" is mislabelled. The complete four-criterion `ROC_AUC` is also computed on the discrete `Modified_SIRS` integer score, not a probability.

---

### E-05 — GRU sequence model

| Field | Detail |
|---|---|
| Notebook | `notebooks/08_gru.ipynb` |
| Model file | `models/gru_baseline.pt` (230,809 B, tracked) |
| Result files | `results/gru_validation_metrics.csv`, `gru_test_metrics.csv`, `gru_test_confusion_matrix.csv`, `gru_training_history.csv`, `gru_validation_predictions.csv` |
| Task | Binary hourly sepsis classification from a patient sequence |
| Execution context | macOS, Python 3.12.10, MPS; hard-coded `/Users/...` paths in the notebook |

**Preprocessing [V]**

1. 34 missingness indicators for measured vitals/labs, computed **before** imputation.
2. Causal forward fill within each patient (no backward fill).
3. Remaining missing values filled with **training-set medians**.
4. Standardised with **training-set** means/stds; clipped to ±10.
5. Input per hour: **74 features** = 40 values + 34 flags.

Statistics are computed from training patients only. The scaler/medians are stored inside the checkpoint, not as separate artifacts.

**Architecture[V]**

| Component | Value |
|---|---|
| Recurrent layer | 2-layer GRU, hidden 64, dropout 0.2, `batch_first=True` |
| Head | Linear 64→64, ReLU, Dropout 0.2, Linear 64→1 |
| Parameters | 56,065 |
| Output | One logit per hour (many-to-many), not one label per patient |

**Training [V]**

| Parameter | Value |
|---|---|
| Loss | `BCEWithLogitsLoss(pos_weight=54.54, reduction="none")`, masked mean over real hours |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-5 |
| Scheduler | `ReduceLROnPlateau(mode="max", factor=0.5, patience=1)` |
| Batch size | 64 train / 256 eval, bucket sampler by length |
| Max epochs | 30 |
| Early-stopping patience | 4 |
| Gradient clipping | 1.0 |
| Best epoch | **9** |
| Best validation AUC | **0.789997** |
| Epochs run | 13 (early stop) |

**Validation metrics at threshold 0.5 [V]**

| Metric | Value |
|---|---:|
| ROC-AUC | 0.789997 |
| Accuracy | 0.874041 |
| Precision | 0.067129 |
| Recall | 0.465642 |
| F1 | 0.117341 |
| Confusion matrix | TN 200,378 / FP 26,932 / FN 2,224 / TP 1,938 |

**Test metrics at threshold 0.5 [V]**

| Metric | Value |
|---|---:|
| ROC-AUC | 0.815660 |
| AUPRC | 0.101831 (recomputed from the ensemble prediction file) |
| Accuracy | 0.876698 |
| Precision | 0.078710 |
| Recall | 0.550072 |
| F1 | 0.137715 |
| Confusion matrix | TN 203,105 / FP 27,003 / FN 1,887 / TP 2,307 |

**Caveats**

- The validation split is used for early stopping, so the validation AUC is selected. **[V]**
- Inference batches are padded but **no padding mask is passed** to the GRU. Because the GRU is unidirectional and valid-position outputs do not depend on later padding, this does not corrupt predictions; it only wastes compute. Loss and metrics do use the mask. **[V]**
- The GRU and XGBoost input representations differ (causal fill + flags vs raw NaN rows), so the ensemble gain cannot be attributed cleanly to architecture. **[V]**
- The notebook writes results to macOS paths, but the metric CSVs are present in this repository. The model checkpoint is tracked. **[V]**

---

### E-06 — LSTM sequence model

| Field | Detail |
|---|---|
| Notebook | `notebooks/08_lstm.ipynb` |
| Model file | `models/lstm_epoch5.pt` (201,428 B, tracked) |
| Result files | `results/lstm_test_metrics.csv`, `lstm_test_metrics.json`, `lstm_validation_thresholds.csv` |
| Task | Binary hourly sepsis classification from a patient sequence |
| Execution context | Windows (`C:\Users\keert\...`), Python 3.13.7, CPU; old `archive (2)` layout |

**Preprocessing [V]**

- 40 raw feature columns.
- `replace(inf, -inf → NaN)`; then `ffill().bfill().fillna(0)` **within each patient**.
- No scaling, no missingness indicators, no clipping, no training-only statistics.
- **`bfill()` is a temporal-leakage concern:** a leading missing hour can be filled from a later observation in the same patient. It is not outcome leakage, but it violates strict causality and differs from the GRU's forward-only fill. **[V] for code; [S] for magnitude** — the effect was not quantified.

**Architecture [V]**

| Component | Value |
|---|---|
| Recurrent layer | 1-layer LSTM, input 40, hidden 64, `batch_first=True` |
| Regularisation | Dropout 0.2 on the output sequence |
| Head | Linear 64→1 |
| Output | One logit per hour |

**Full-cohort training run [V]**

| Parameter | Value |
|---|---|
| Loss | `BCEWithLogitsLoss(pos_weight≈54.60)` |
| Optimizer | Adam, lr 0.001 |
| Batch size | 32 |
| Epochs | 5 (fixed; no early stopping, no best-checkpoint selection) |
| Validation AUC by epoch | 0.7603, 0.7679, 0.7673, 0.7745, **0.7825** |
| Saved checkpoint | End of epoch 5 |

The positive weight was computed over **all** patient files (27,916 positives / 1,524,294 negatives), not training patients only. **[V]**

**Test metrics at threshold 0.5 [V]**

| Metric | Value |
|---|---:|
| ROC-AUC | 0.793767 |
| AUPRC | not saved (no test probability file) |
| Accuracy | 0.907316 |
| Precision | 0.093985 |
| Recall | 0.483548 |
| F1 | 0.157380 |
| Confusion matrix | TN 210,558 / FP 19,550 / FN 2,166 / TP 2,028 |

**Abandoned toy run — do not cite [V]**

The notebook contains an earlier 10-epoch run on an **80-patient / 3,078-timestep** training tensor and a 626-observation test subset. It reported test AUC **0.5569**, precision 0, recall 0, F1 0 at threshold 0.5. These numbers are artefacts of the toy subset and are **not** the LSTM result. The saved LSTM metrics correspond to the later full 5-epoch run.

**Other caveats**

- The notebook contains failed `pip install` cells, `ModuleNotFoundError` cells, a `NameError` on `X_train_pad`, and duplicate/abandoned cells. It is not a clean reproduction script. **[V]**
- The checkpoint was saved under the `keert` project path; the copy in this repository is tracked, but the notebook's paths do not resolve here. **[V]**
- `torch` is absent from `.venv`, so the checkpoint cannot be loaded in the documented environment. **[V]**

---

### E-07 — Transformer sequence model

| Field | Detail |
|---|---|
| Notebook | `notebooks/08_transformer.ipynb` |
| Model file | `models/transformer_best.pt` (297,555 B, tracked) |
| Result files | `results/transformer_test_metrics_padding_mask.csv`, `transformer_test_confusion_matrix_padding_mask.csv`, `transformer_training_history.csv`, `transformer_training_loss.png`, `transformer_test_confusion_matrix.png` |
| Task | Binary hourly sepsis classification from a 24-hour window |
| Execution context | `C:\TwinSepsis`, Python 3.13.5, CPU |

**Preprocessing [V]**

- 40 raw features.
- `SimpleImputer(strategy="median")` fitted on training rows; then `StandardScaler` fitted on training rows; then `nan_to_num`.
- No forward fill and no missingness indicators.
- Sequences are **causal windows ending at the current hour**: for hour `t`, the model sees up to the previous 24 hours including `t`. Shorter stays are left-padded, and a padding mask marks artificial positions.
- The imputer, scaler, and feature list are written by the notebook to `models/transformer_imputer.joblib`, `transformer_scaler.joblib`, and `transformer_features.json`, but **those files are absent**. **[V]**
- **Reproducibility consequence:** even with `torch` installed, test preprocessing cannot be reproduced exactly without refitting the imputer/scaler on the training rows (which may or may not reproduce the saved numbers). **[S/V]**

**Architecture [V]**

| Component | Value |
|---|---|
| Input projection | Linear 40→64 |
| Positional encoding | Learned embedding, length 24 |
| Encoder | 2 `TransformerEncoderLayer`s |
| Heads | 4 |
| Feed-forward | 128 |
| Dropout | 0.1 |
| Output | Linear to 1 logit per window |
| Sequence length | 24 |
| Batch size | 64 |

**Training [V]**

| Parameter | Value |
|---|---|
| Loss | `BCEWithLogitsLoss(pos_weight=54.54)` |
| Optimizer | Adam, lr 1e-3, weight decay 1e-4 |
| Max epochs | 5 |
| Early-stopping patience | 2 |
| Best epoch | **1** |
| Best validation AUC | **0.728275** |
| Epochs run | 3 (early stop) |

Validation history:

| Epoch | Train loss | Val AUC | Val AUPRC |
|---:|---:|---:|---:|
| 1 | 1.4161 | **0.7283** | 0.0668 |
| 2 | 1.3859 | 0.6922 | 0.0572 |
| 3 | 1.3662 | 0.7173 | 0.0597 |

**Test metrics at threshold 0.5 [V]**

| Metric | Value |
|---|---:|
| ROC-AUC | 0.758029 |
| AUPRC | 0.081068 |
| Accuracy | 0.930466 |
| Precision | 0.105414 |
| Recall | 0.385312 |
| F1 | 0.165540 |
| Confusion matrix | TN 216,394 / FP 13,714 / FN 2,578 / TP 1,616 |

**Caveats**

- The corrected results are saved with the `_padding_mask` suffix. The notebook was fixed in commit `b2c4add`; the older uncorrected metrics are not present in `results/`. **[V]**
- Best epoch 1 with early stop after epoch 3 suggests unstable optimisation or overfitting; the model is the weakest of the three neural architectures by AUC. **[V] for metrics; [S] for the cause.**
- `torch` absent and preprocessing artifacts missing → not reproducible now. **[V]**

---

### E-08 — Weighted XGBoost + GRU ensemble

| Field | Detail |
|---|---|
| Notebook | `notebooks/09_xgboost_gru_ensemble.ipynb` |
| Result files | `ensemble_config.json`, `ensemble_final_test_comparison.csv`, `ensemble_validation_comparison.csv`, `ensemble_validation_weight_search.csv`, `ensemble_validation_threshold_search.csv`, `ensemble_validation_threshold_comparison.csv`, `ensemble_test_predictions.csv`, three per-model test confusion CSVs |
| Inputs | Saved validation/test probabilities from XGBoost and GRU; no retraining |
| Task | Combine two hourly probability streams, then classify |

**Alignment [V]**

- Merge key: `(patient_id, hour_index)`, `how="outer"`, one-to-one validation.
- Validation: 231,472 GRU rows, 231,472 XGBoost rows, 231,472 matched, 0 unmatched; labels identical.
- Test: 234,302 rows, 4,194 positive hours, 6,051 patients.

**Weight selection [V]**

- Grid over XGBoost weight `w` (GRU weight `1-w`).
- Selection metric: **validation ROC-AUC**.
- Selected: **XGBoost 0.55 / GRU 0.45**.
- Validation AUC 0.812665, AUPRC 0.085824.
- The top three weights are extremely close: 0.55/0.45 → 0.812665; 0.60/0.40 → 0.812551; 0.50/0.50 → 0.812526. The choice is not robust. **[V] for values; [S] for instability.**

**Threshold selection [V]**

- Grid `0.01 … 0.95` in steps of 0.01.
- Selection metric: **validation F1**.
- Selected: **0.81**.
- Validation F1 surface around the choice: 0.78 → 0.15695, 0.79 → 0.15743, 0.80 → 0.15748, **0.81 → 0.15870**, 0.82 → 0.15700, 0.83 → 0.15607, 0.84 → 0.15697, 0.85 → 0.15799, 0.86 → 0.15848.
- The surface is nearly flat, so 0.81 is a validation artefact rather than a stable clinical operating point. **[V] for values; [S] for interpretation.**

**Validation metrics at threshold 0.81 [V]**

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| XGBoost | 0.114625 | 0.243873 | 0.155950 |
| GRU | 0.087849 | 0.358722 | 0.141135 |
| Weighted ensemble | 0.110370 | 0.282316 | 0.158698 |

Validation ROC-AUC / AUPRC (threshold-free): XGBoost 0.796669 / 0.082692; GRU 0.789997 / 0.074066; ensemble 0.812665 / 0.085824. **[V]**

**Test metrics at threshold 0.81 [V]**

| Model | ROC-AUC | AUPRC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.816563 | 0.121752 | 0.952924 | 0.139680 | 0.315928 | 0.193713 |
| GRU | 0.815660 | 0.101831 | 0.921644 | 0.101547 | 0.430377 | 0.164322 |
| **Weighted ensemble** | **0.838431** | **0.131937** | **0.946065** | **0.128814** | **0.349309** | **0.188219** |

All three rows use the same 234,302 test rows and 4,194 positive hours. Ensemble CM: TN 216,874 / FP 23,236 / FN 2,729 / TP 1,463 (derived from the saved prediction file; the CSV in `results/ensemble_ensemble_test_confusion_matrix.csv` is consistent).

**Reproduction check [V]**

Recomputing AUROC/AUPRC/precision/recall/F1 at 0.81 directly from `results/ensemble_test_predictions.csv` reproduced `results/ensemble_final_test_comparison.csv` to all printed digits. This is the only experiment whose headline metrics were independently re-derived from saved row-level predictions.

**Caveats**

- **Validation is used three times in this experiment alone:** XGBoost early stopping, ensemble weight selection, threshold selection. Validation metrics are therefore optimistic. **[V]**
- `ensemble_config.json` still says `"test_evaluated": false` even though the test comparison exists. The flag is stale provenance, not evidence that the test was not run. **[V]**
- The ensemble improves AUC and AUPRC but not precision/F1 relative to XGBoost at 0.81 (XGBoost F1 0.1937 vs ensemble 0.1882). The claim "ensemble is best" is true for ranking metrics, not for F1 at this threshold. **[V]**
- The XGBoost and GRU rows in the ensemble test table use threshold 0.81, whereas their own metric files use 0.5. They are **not** the same operating points. **[V]**
- `ensemble_test_predictions.csv` contains `patient_id` and is committed. This is record-level, though not PHI. **[V]**

---

## 4. Cross-model comparison (read the threshold column)

All rows are hourly-row metrics on the same 6,051-patient test split. **Every row must be read with its own threshold.** A "best F1" comparison across rows is invalid because the thresholds differ.

| Model | Threshold | ROC-AUC | AUPRC | Precision | Recall | F1 | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| XGBoost | 0.50 | 0.816563 | 0.121752* | 0.061933 | 0.662136 | 0.113271 | `xgboost_test_metrics.csv` |
| XGBoost | 0.81 | 0.816563 | 0.121752* | 0.139680 | 0.315928 | 0.193713 | `ensemble_final_test_comparison.csv` |
| Random Forest | 0.17 | 0.781508 | n/s | 0.090812 | 0.323796 | 0.141842 | `random_forest_test_metrics.csv` |
| GRU | 0.50 | 0.815660 | 0.101831* | 0.078710 | 0.550072 | 0.137715 | `gru_test_metrics.csv` |
| GRU | 0.81 | 0.815660 | 0.101831* | 0.101547 | 0.430377 | 0.164322 | `ensemble_final_test_comparison.csv` |
| LSTM | 0.50 | 0.793767 | n/s | 0.093985 | 0.483548 | 0.157380 | `lstm_test_metrics.csv` |
| Transformer | 0.50 | 0.758029 | 0.081068 | 0.105414 | 0.385312 | 0.165540 | `transformer_test_metrics_padding_mask.csv` |
| **Weighted ensemble** | **0.81** | **0.838431** | **0.131937** | **0.128814** | **0.349309** | **0.188219** | `ensemble_final_test_comparison.csv` |
| Partial SOFA (any component) | 2 points | n/s | n/s | 0.0333† | 0.0466† | 0.0389† | notebook CM, validation |
| Modified SIRS (any criterion) | 2 points | n/s | n/s | 0.0343† | 0.2947† | 0.0614† | notebook CM, validation |

`n/s` = not saved. `*` = AUPRC recomputed from `ensemble_test_predictions.csv`; the per-model CSV does not contain AUPRC. `†` = derived from saved confusion matrix, validation split, not test.

**What the table supports [V]**

- The weighted ensemble has the best test AUC (0.8384) and AUPRC (0.1319).
- XGBoost has the best test F1 at its 0.81 operating point (0.1937), slightly above the ensemble (0.1882).
- XGBoost has the best single-model test AUC (0.8166) and AUPRC (0.1218).
- All accuracies are dominated by the 98.2% negative class and are not evidence of clinical utility.

**What the table does not support**

- It does **not** support a claim that any model is clinically deployable.
- It does **not** support comparing F1 across rows at different thresholds.
- It does **not** include a patient-level metric, the official PhysioNet utility score, calibration, confidence intervals, or lead-time analysis.
- It does **not** include a test evaluation for SOFA or SIRS.

---

## 5. Selection chronology and validation reuse

| Step | What was selected | Data used | Evidence |
|---|---|---|---|
| Split | Patient-level train/val/test IDs | Unknown generator | ID files only; notebook 03 empty |
| XGBoost | Number of boosting rounds (196) | Validation AUC | Early stopping |
| Random Forest | Decision threshold (0.17) | Validation F1 | Threshold sweep |
| GRU | Best epoch (9) | Validation AUC | Early stopping |
| LSTM | Final epoch (5) | Validation AUC monitored only | Fixed 5 epochs |
| Transformer | Best epoch (1) | Validation AUC | Early stopping |
| Ensemble | Weights (0.55/0.45) | Validation AUC | Weight grid |
| Ensemble | Threshold (0.81) | Validation F1 | Threshold grid |
| Final test | None | Test split | Frozen choices, evaluated once |

**Consequence [V]:** the validation split is not an unbiased estimate for any model that used it for early stopping or tuning, and the ensemble's validation metrics are the most heavily selected. The test split is methodologically acceptable as a held-out set because choices were frozen before test evaluation, but the study still lacks a clean "tune" split separate from "validation". See F-05.

---

## 6. Artifact provenance

| Artifact | Produced by | Exists | Tracked | Notes |
|---|---|---|---|---|
| `models/xgboost_baseline.json` | 04 | Yes | Yes | 1.54 MB; loadable with installed `xgboost` |
| `models/random_forest_baseline.joblib` | 05 | Yes | **No** | 1.04 GB; git-ignored by `models/*.joblib` |
| `models/gru_baseline.pt` | 08_gru | Yes | Yes | 230,809 B; `torch` absent locally |
| `models/lstm_epoch5.pt` | 08_lstm | Yes | Yes | 201,428 B; `torch` absent locally |
| `models/transformer_best.pt` | 08_transformer | Yes | Yes | 297,555 B; `torch` absent locally |
| `models/transformer_imputer.joblib` | 08_transformer | **No** | — | Written by notebook; absent |
| `models/transformer_scaler.joblib` | 08_transformer | **No** | — | Written by notebook; absent |
| `models/transformer_features.json` | 08_transformer | **No** | — | Written by notebook; absent |
| `results/xgboost_validation_metrics.csv` | 04 | Yes | Yes | threshold 0.5 |
| `results/xgboost_test_metrics.csv` | 04 | Yes | Yes | threshold 0.5 |
| `results/random_forest_test_metrics.csv` | 05 | Yes | Yes | threshold 0.17 |
| `results/gru_validation_metrics.csv` | 08_gru | Yes | Yes | threshold 0.5 |
| `results/gru_test_metrics.csv` | 08_gru | Yes | Yes | threshold 0.5 |
| `results/gru_training_history.csv` | 08_gru | Yes | Yes | 13 epochs |
| `results/gru_validation_predictions.csv` | 08_gru / 09 | Yes | Yes | row-level probabilities |
| `results/lstm_test_metrics.csv` and `.json` | 08_lstm | Yes | Yes | full 5-epoch run |
| `results/lstm_validation_thresholds.csv` | 08_lstm | Yes | Yes | threshold sweep |
| `results/transformer_test_metrics_padding_mask.csv` | 08_transformer | Yes | Yes | corrected run |
| `results/transformer_training_history.csv` | 08_transformer | Yes | Yes | 3 epochs |
| `results/ensemble_config.json` | 09 | Yes | Yes | **stale `test_evaluated: false`** |
| `results/ensemble_final_test_comparison.csv` | 09 | Yes | Yes | threshold 0.81 |
| `results/ensemble_test_predictions.csv` | 09 | Yes | Yes | 234,302 rows; 14.3 MB; record-level |
| SOFA / SIRS metrics | 06 / 07 | **No** | — | only notebook outputs; no CSV/JSON |

**Important provenance gap:** there is no committed script that regenerates `results/` from raw data. Every notebook re-implements data loading and preprocessing inline, and notebook 03 (the split generator) is empty. The ensemble is the exception: its metrics can be regenerated from saved prediction probabilities without retraining.

---

## 7. Reproducibility matrix

| Experiment | Code present | Data present | Weights present | Preprocessing reproducible | End-to-end from a clean clone? |
|---|---|---|---|---|---|
| EDA | Yes | Yes | n/a | n/a | **Yes** (not a model) |
| XGBoost | Yes | Yes | Yes (JSON) | Stateless / raw rows | **Yes**, modulo the missing split generator |
| Random Forest | Yes | Yes | Local only (untracked 1.04 GB) | Stateless / raw rows | **No** — weights not in git |
| Partial SOFA | Yes | Yes | n/a | Inline, deterministic | **Runs**, but no result artifact and not canonical |
| Modified SIRS | Yes | Yes | n/a | Inline, deterministic | **Runs**, but no result artifact and not canonical |
| GRU | Yes | Yes | Yes | Stats inside checkpoint, but notebook uses macOS paths and missing cached pickles | **No** — `torch` absent, paths broken |
| LSTM | Messy | Yes | Yes | Inline, not persisted; uses `bfill()` | **No** — `torch` absent, old layout, broken cells |
| Transformer | Yes | Yes | Yes | Imputer/scaler/features missing | **No** — `torch` absent and preprocessing artifacts absent |
| Ensemble | Yes | Yes (saved predictions) | Derived | Frozen config | **Yes** for metrics; training not required |

Legend: "clean clone" means the 52 tracked files plus git-ignored datasets, without untracked 1 GB models. `torch` absence alone makes GRU/LSTM/Transformer non-reproducible in the documented environment.

---

## 8. Metric-quality issues (do not smooth these over)

1. **Hour-level only.** A long ICU stay contributes up to 336 rows; the effective sample size for patient-level inference is 6,051, not 234,302. No patient-level AUROC/AUPRC is reported. **[V]**
2. **No official PhysioNet metric.** The Challenge 2019 utility score (which rewards early detection and penalises late/false alarms at the patient level) was **not computed** anywhere in the repository. No code for it exists. **[V]**
3. **No calibration.** Raw probabilities are thresholded at 0.81 in the ensemble, far from 0.5, and linearly averaged without calibration. No Brier score, reliability diagram, or calibration slope. **[V]**
4. **No uncertainty.** No bootstrap confidence intervals, no repeated splits, no cross-validation. **[V]**
5. **No lead-time analysis.** The notebooks never stratify by time-to-onset, so "early" vs "late" detection cannot be distinguished. **[V]**
6. **Threshold heterogeneity.** 0.5 (XGBoost, GRU, LSTM, Transformer), 0.17 (Random Forest), 0.81 (ensemble table). The saved files are not a like-for-like comparison. **[V]**
7. **Accuracy is misleading.** At 1.8% prevalence, an all-negative classifier scores 98.2% accuracy. Several reported accuracies are near that floor while recall is low (e.g., Random Forest at 0.5: accuracy 0.9803, recall 0.0084). **[V]**
8. **AUPRC is missing for most models.** Only Transformer and the ensemble table save AUPRC; XGBoost/GRU AUPRC in this document were recomputed from the ensemble prediction file, and Random Forest/LSTM have no saved test probabilities. **[V]**
9. **SOFA/SIRS "AUC" is partly not AUC.** The SIRS component metric is computed on binary scores; the SOFA partial score has no saved AUC at all. **[V]**
10. **The LSTM notebook contains a misleading toy run.** Its 0.5569 AUC must not be quoted as the LSTM result. **[V]**
11. **The ensemble config is stale.** `test_evaluated: false` contradicts the existence of the test comparison. **[V]**
12. **No statistical test compares models.** The ensemble's AUC advantage over XGBoost (0.8384 vs 0.8166) is reported without a confidence interval or paired test. **[V]**

---

## 9. Bottom line

- The repository contains a **complete PhysioNet 2019 study**, not a MIMIC-IV study and not a digital twin.
- The **best ranking model** on the held-out test split is the weighted XGBoost+GRU ensemble: AUC 0.8384, AUPRC 0.1319 at threshold 0.81.
- The **best single model** by test AUC/AUPRC is XGBoost: 0.8166 / 0.1218.
- The **best test F1 at a tuned threshold** is XGBoost at 0.81: 0.1937, marginally above the ensemble's 0.1882.
- All headline numbers are **hour-level, uncalibrated, without confidence intervals, and without the official PhysioNet utility score**.
- The clinical-score experiments are exploratory and non-canonical; they must not be used as Sepsis-3 labels or as clinical baselines.
- For the new MIMIC-IV work, the defensible starting point is **not** to port these notebooks. It is to build a separate, auditable pipeline with a committed split generator, a canonical Sepsis-3 label, causal preprocessing, patient-level metrics, and calibration. See `preprocessing_plan.md`.

---

## 10. Cross-references

| Topic | Where |
|---|---|
| Repository structure, environment, MIMIC inventory | `repository_audit.md` §§3–6 |
| Split provenance | `repository_audit.md` §8, F-01 |
| Reproducibility | `repository_audit.md` §9 |
| Methodological issues | `repository_audit.md` §10 |
| SOFA/SIRS detail | `repository_audit.md` §11 |
| Privacy/governance | `repository_audit.md` §12 |
| Findings register | `repository_audit.md` §13 (F-01 … F-24) |
| MIMIC table schema | `mimic_table_inventory.csv` |
| Proposed MIMIC pipeline | `preprocessing_plan.md` |
