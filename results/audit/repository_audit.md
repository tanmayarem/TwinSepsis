# Repository Audit — TwinSepsis

| Field | Value |
|---|---|
| Audit date | 2026-10-29 |
| Repository | `C:\TwinSepsis` |
| Branch / commit | `main` / `a7659b6` ("Ignore raw MIMIC-IV archive") |
| Working tree | Clean at audit start (no modified, staged, or untracked files reported by `git status --porcelain`) |
| Mode | Read-only. No source, notebook, model, config, dependency, dataset, or result file was created, modified, moved, or deleted. |
| New files | Only the four files in `results/audit/` (this directory). |
| Confidentiality | No raw clinical rows, patient identifiers, or record-level extracts appear in this report. All numbers are aggregates, counts, or already-published metric values. |

Legend: **[V]** = verified directly from the repository (file, output, or computed aggregate). **[S]** = suspected / inferred, with the reasoning stated. Nothing in this report is presented as verified unless it was actually observed.

---

## 1. Executive summary

1. **[V]** The repository contains a complete, working **PhysioNet Challenge 2019** sepsis-prediction study (40,336 patients, 1,552,210 hourly records) with five model families: XGBoost, Random Forest, GRU, LSTM, and Transformer, plus a weighted XGBoost+GRU ensemble.
2. **[V]** The best-performing model on the held-out test split is the **weighted ensemble** (AUC 0.8384, AUPRC 0.1319, F1 0.1882 at threshold 0.81). Its headline numbers were independently reproduced from `results/ensemble_test_predictions.csv`.
3. **[V]** **No MIMIC-IV code exists anywhere in the repository.** The MIMIC-IV dataset (92.7 GB, 31 tables) was added to `dataset/mimic-iv/`, and the commit titled "adding next dataset for forecasting and treatment" changed only `.gitignore`. There is no loader, cohort builder, mapper, label definition, or notebook that reads MIMIC-IV.
4. **[V]** The patient split is undocumented in code: `notebooks/03_preprocessing_and_split.ipynb` contains **zero cells** (129 bytes), yet the split ID files drive every experiment. The split is internally consistent (28,235 / 6,050 / 6,051, no overlap) and appears stratified, but the generation method cannot be reproduced from the repository.
5. **[V]** The environment is **not reproducible as committed**: `requirements.txt` does not list `torch`, and `torch` is not installed in `.venv`. The GRU, LSTM, and Transformer checkpoints therefore cannot be loaded or re-run in the documented environment.
6. **[V]** The GRU notebook was executed on **macOS** with hard-coded `/Users/...` paths and depends on a cached `dataset/processed/gru_raw_*.pkl` that does not exist locally. The LSTM notebook was executed on a different Windows machine (`C:\Users\keert\...`) and points at an `archive (2)` folder layout that is not the current dataset layout.
7. **[V]** The **validation split is used three times** — early stopping for XGBoost, ensemble weight selection, and threshold selection — so validation metrics are optimistic. The test split is evaluated once with frozen choices, which is methodologically acceptable, but it is not a clean untouched validation.
8. **[V]** The clinical-score notebooks are **partial and non-canonical**: SOFA uses 4 of 6 components (no respiratory, no CNS, MAP-only cardiovascular, creatinine-only renal) and has two boundary errors; SIRS is a modified 4-criterion score. The SOFA notebook itself states canonical SOFA cannot be reproduced. These scores must not be described as SOFA/SIRS in publications or in any new pipeline.
9. **[V]** Reporting is **hour-level only**. No patient-level AUROC/AUPRC, no PhysioNet official utility score, no calibration analysis, no time-to-onset stratification, and no reported confidence intervals. XGBoost, GRU, Random Forest, Transformer, and ensemble metrics in `results/` are computed at four different thresholds (0.5, 0.5, 0.17, 0.5, 0.81), so the saved files are not directly comparable as a table.
10. **[V]** There is **no digital-twin, agent/LLM, or Streamlit layer** in the repository, and no dependency for any of them. The repository is currently a classical/sequence-model sepsis prediction study, not a twin or agentic system.

---

## 2. Scope, method, and constraints

### 2.1 What was inspected

| Area | Method |
|---|---|
| Git state and history | `git log`, `git show --stat`, `git show <commit> -- .gitignore`, `git branch -a`, `git remote -v`, `git ls-files`, `git grep` |
| Notebooks | Extracted cell source and stored outputs to a temporary directory outside the repository; read as text. Notebooks were **not executed**. |
| Source | Direct read of `src/config.py` and `src/clinical_score_plots.py` |
| Results | Read all small CSV/JSON artifacts; recomputed ensemble metrics from the saved prediction CSV |
| splits | Counts, duplicates, pairwise overlaps, union, and cross-tabulation against `patient_summary.csv` |
| PhysioNet data | Directory counts and file sizes only |
| MIMIC-IV data | Header line (row 1) of all 31 CSVs; aggregate scans of `icustays`, `admissions`, `patients`; dictionary lookups in `d_items` and `d_labitems` |
| Environment | `python -c "importlib.util.find_spec(...)"` for key packages; `requirements.txt` review |

### 2.2 What was deliberately not done

- No notebook execution, no model training, no inference re-run.
- No full load of `chartevents` (≈40 GB), `labevents` (≈17.5 GB), `emar`, `emar_detail`, `poe`, or `pharmacy`.
- No dataset download, extraction, decompression, or checksum verification.
- No file modification outside `results/audit/`.
- No commit, push, branch switch, or remote operation.

### 2.3 Impact of constraints

Because notebooks were read from stored outputs rather than re-executed, the audit distinguishes **"the notebook reports X"** from **"X is reproducible"**. Only the ensemble test metrics were independently recomputed. All other reported metrics are treated as claims recorded in notebook outputs or result files.

---

## 3. Repository structure

```
C:\TwinSepsis
├── .env                     (PHYSIONET_ROOT only; value redacted here)
├── .gitignore
├── requirements.txt         (pip freeze, 128 pinned packages)
├── archive.zip              (78 MB, git-ignored)
├── dataset/                 (git-ignored)
│   ├── mimic-iv/            (31 tables, 92.7 GB)
│   └── physionet.org/       (Challenge 2019, 391.8 MB)
├── models/                  (5 files, 4 tracked)
├── notebooks/               (11 notebooks)
├── results/                 (33 files, 28.7 MB, all tracked)
└── src/
    ├── config.py
    └── clinical_score_plots.py
```

- **52 tracked files** in total.
- No `README`, no `LICENSE` for the project, no `pyproject.toml`, no `setup.py`, no `Makefile`, no CI configuration, no tests, no `src/__init__.py`.
- `src/__pycache__/` exists locally and is git-ignored.

---

## 4. Environment and dependency audit

| Item | Finding |
|---|---|
| Interpreter | System and `.venv` are both **Python 3.14.7** (`C:\TwinSepsis\.venv\Scripts\python.exe`) **[V]** |
| `requirements.txt` | 128 pinned packages, including `xgboost==3.4.1`, `shap==0.52.0`, `scikit-learn==1.9.1`, `pandas==3.0.6`, `numpy==2.5.3`, `pyarrow==25.0.1`, full Jupyter stack **[V]** |
| `torch` | **Not listed** in `requirements.txt`; **not installed** in `.venv` (`find_spec` → None) **[V]** |
| Other absent packages | `streamlit`, `openai`, `transformers`, `langchain`, any LLM/agent SDK **[V]** |
| Notebook kernels | `01`, `02` → `.venv (3.14.7)`; `08_gru` → Python 3.12.10 on macOS (MPS); `08_lstm` → Python 3.13.7 (CPU, `keert` machine); `08_transformer` → 3.13.5 (`.venv`, `C:\TwinSepsis`) **[V]** |
| Consequence | Three architectures cannot be re-trained or even loaded in the committed environment. Re-running them requires adding `torch` (a dependency change that is out of scope for this audit). |

`src/config.py` **[V]**:

- Reads `PHYSIONET_ROOT` from `.env` via `python-dotenv`.
- Calls `Path(os.getenv("PHYSIONET_ROOT"))` with no guard: importing the module fails with `TypeError` if `.env` is missing or the variable is unset.
- Creates `results/` and `models/` as an import side effect.
- Defines only PhysioNet paths. There is no MIMIC-IV configuration surface.

---

## 5. Data inventory

### 5.1 PhysioNet Challenge 2019 (the only dataset actually used)

| Metric | Value | Status |
|---|---|---|
| Patient files | 40,336 (setA 20,336 + setB 20,000) | **[V]** |
| Columns per file | 41 (40 features + `SepsisLabel`) | **[V]** |
| Total hourly records | 1,552,210 | **[V]** |
| Sepsis-positive hours | 27,916 (1.798%) | **[V]** |
| Patients with any positive hour | 2,932 (7.269%) | **[V]** |
| setA hours / septic patients | 790,215 / 1,790 | **[V]** |
| setB hours / septic patients | 761,995 / 1,142 | **[V]** |
| Extra files | `Dataset.csv` (146.7 MB), `LICENSE.txt`, manuscript PDF, `SHA256SUMS.txt`, two utility SVGs | **[V]** |
| Local storage | 391.8 MB | **[V]** |

Feature families: 8 vitals, 26 labs, 6 demographics/admin (`Age`, `Gender`, `Unit1`, `Unit2`, `HospAdmTime`, `ICULOS`). Exact order is stable across all 40,336 files.

### 5.2 MIMIC-IV (present but unused)

| Metric | Value | Status |
|---|---|---|
| Tables | 31 (22 `hosp` + 9 `icu`) | **[V]** |
| Storage | 92,688 MB (≈92.7 GB) | **[V]** |
| ICU stays | 94,458 | **[V]** |
| Distinct ICU subjects | 65,366 | **[V]** |
| Distinct admissions with an ICU stay | 85,242 | **[V]** |
| Hospital admissions (all) | 546,028 | **[V]** |
| Patients (all) | 364,627 | **[V]** |
| ICU LOS (days) | median 1.97, mean 3.63, p95 12.68, max 226.4 | **[V]** |
| In-hospital mortality (`hospital_expire_flag`) | 2.16% of admissions | **[V]** |
| Date range in `icustays` | 2110-01-11 → 2214-07-26 (shifted dates) | **[V]** |
| Anchor year groups | 2008-2010 … 2020-2022 | **[V]** |
| Dataset version | Not recorded anywhere in the repo. Range is consistent with MIMIC-IV v3.1 (2008–2022), but this is **[S]** and must be confirmed from the source package. | |

Full column-level inventory: `results/audit/mimic_table_inventory.csv`.

### 5.3 Storage-layout anomalies (verified)

1. **[V] Double-nested table paths.** Every MIMIC table is stored as `dataset/mimic-iv/<module>/<table>.csv/<table>.csv`, where `<table>.csv` is a **directory**. This is almost certainly a side effect of extracting `hosp/<table>.csv.gz` with a tool that mistook the member name for a directory name. Any new loader must either handle this layout explicitly or be given a corrected path — and correcting it would mean renaming directories, which is out of scope here.
2. **[V] Checksums do not match the files on disk.** `SHA256SUMS.txt` lists `hosp/<table>.csv.gz` and `icu/<table>.csv.gz`; the files on disk are uncompressed `.csv`. The shipped checksums therefore cannot validate the local CSVs without the original archives.
3. **[V] Missing root metadata.** `SHA256SUMS.txt` lists `CHANGELOG.txt` and `LICENSE.txt`, but neither exists at `dataset/mimic-iv/`. Only `hosp/`, `icu/`, and `SHA256SUMS.txt` are present.
4. **[V] `dataset/processed/` does not exist.** The GRU notebook's cached pickles (`gru_raw_train.pkl`, etc.) are absent.
5. **[V] `archive.zip` (78 MB)** sits at the repository root and is git-ignored. Its contents were not inspected.

---

## 6. Code inventory

### 6.1 Notebooks

| Notebook | Cells | Purpose | Last verified execution context |
|---|---|---|---|
| `01_data_understanding.ipynb` | 19 | Package check, path check, single-patient schema inspection | `C:\TwinSepsis`, Python 3.14.7 |
| `02_full_dataset_eda.ipynb` | 15 | Full-cohort EDA; writes `results/patient_summary.csv` | `C:\TwinSepsis`, Python 3.14.7 |
| `03_preprocessing_and_split.ipynb` | **0** | **Empty stub (129 bytes)** — split provenance missing | n/a |
| `04_xgboost.ipynb` | — | XGBoost baseline | `C:\TwinSepsis` |
| `05_random_forest.ipynb` | 14 | Random Forest baseline | `C:\TwinSepsis` |
| `06_sofa.ipynb` | — | Partial SOFA rule-based analysis | `C:\TwinSepsis` |
| `07_sirs.ipynb` | — | Modified SIRS rule-based analysis | `C:\TwinSepsis` |
| `08_gru.ipynb` | — | GRU sequence model + training/eval | macOS, Python 3.12.10, MPS |
| `08_lstm.ipynb` | 77 | LSTM sequence model (contains abandoned cells) | Windows (`keert`), Python 3.13.7, CPU |
| `08_transformer.ipynb` | 13 | Transformer with padding mask | `C:\TwinSepsis`, Python 3.13.5, CPU |
| `09_xgboost_gru_ensemble.ipynb` | — | Weighted ensemble, threshold selection, test evaluation | `C:\TwinSepsis` |

### 6.2 Source modules

| File | Role |
|---|---|
| `src/config.py` | PhysioNet paths, results/models directories |
| `src/clinical_score_plots.py` | `plot_score_summary()` — one reusable presentation helper used by SOFA/SIRS |

There is **no** reusable data-loading, preprocessing, feature-engineering, splitting, evaluation, or inference module. Every notebook re-implements patient loading and preprocessing inline. This is the single largest structural weakness for adding a second dataset.

### 6.3 Model artifacts

| Artifact | Size | Tracked | Loadable today |
|---|---|---|---|
| `models/xgboost_baseline.json` | 1.54 MB | Yes | Yes (`xgboost` installed) |
| `models/random_forest_baseline.joblib` | 1.04 GB | **No** (git-ignored by `models/*.joblib`) | Yes (`joblib`/`sklearn` installed) |
| `models/gru_baseline.pt` | 230,809 B | Yes | **No** (`torch` absent) |
| `models/lstm_epoch5.pt` | 201,428 B | Yes | **No** (`torch` absent) |
| `models/transformer_best.pt` | 297,555 B | Yes | **No** (`torch` absent) |
| `models/transformer_imputer.joblib` | — | Absent | Notebook writes it; file not present |
| `models/transformer_scaler.joblib` | — | Absent | Notebook writes it; file not present |
| `models/transformer_features.json` | — | Absent | Notebook writes it; file not present |

---

## 7. What has actually been built

| Workstream | Status | Evidence |
|---|---|---|
| PhysioNet EDA | Complete | `02_full_dataset_eda`, `results/patient_summary.csv` |
| Patient split | **Artifacts exist, generator missing** | 3 ID CSVs; empty notebook 03 |
| XGBoost baseline | Complete | Notebook 04, metrics + confusion matrix + model JSON |
| Random Forest baseline | Complete | Notebook 05, metrics + CM + 1 GB model |
| Partial SOFA | Exploratory only | Notebook 06; no result files saved |
| Modified SIRS | Exploratory only | Notebook 07; no result files saved |
| GRU | Complete | Notebook 08, checkpoint, metrics, history, val predictions |
| LSTM | Complete but messy | Notebook 08, checkpoint, metrics, val thresholds |
| Transformer | Complete | Notebook 08, checkpoint, metrics, history, plots |
| Weighted ensemble | Complete | Notebook 09, config, 8 result files, 14 MB test predictions |
| MIMIC-IV pipeline | **Does not exist** | No code reference to MIMIC anywhere (`git grep -i mimic` matches only `.gitignore`) |
| Digital twin / agentic / LLM / UI | **Does not exist** | No code, no dependencies |
| Tests / CI / packaging | **Does not exist** | No test files, no CI config |

---

## 8. Split provenance audit

**[V] Observed facts**

- `train_patient_ids.csv` = 28,235 IDs, `val_patient_ids.csv` = 6,050, `test_patient_ids.csv` = 6,051.
- Zero duplicates inside each file; zero overlap between any pair; union = 40,336 = total patient count; every ID has a matching `.psv`.
- Cross-tab against `patient_summary.csv`:

| Split | setA | setB | Septic patients | Sepsis rate |
|---|---|---|---|---|
| train | 14,218 | 14,017 | 2,052 | 7.27% |
| val | 3,087 | 2,963 | 440 | 7.27% |
| test | 3,031 | 3,020 | 440 | 7.27% |

**[S] Inference:** the equal sepsis rate to two decimal places in all three splits, together with the near-50/50 setA/setB mix, is strong evidence of a **stratified random patient-level split** (likely stratified on patient-level sepsis status and seeded). This cannot be confirmed because the generating code is absent.

**Risk:** if any future work needs to re-derive the split, it cannot be done from the repo. Any new MIMIC-IV split must be generated by code that is committed and versioned.

---

## 9. Reproducibility audit

| Experiment | Code | Data | Weights | Preprocessing | End-to-end reproducible now? |
|---|---|---|---|---|---|
| EDA | Yes | Yes | n/a | n/a | Yes |
| XGBoost | Yes | Yes | Yes (JSON) | Inline, stateless (raw rows, NaNs kept) | **Yes** (modulo split generator) |
| Random Forest | Yes | Yes | 1 GB joblib present but untracked | Inline, stateless | Locally yes; from git **no** |
| GRU | Yes | Yes | Yes | Stats serialized inside checkpoint | **No** — `torch` missing; notebook has macOS paths |
| LSTM | Partly (many broken cells) | Yes | Yes | Inline; no scaler/imputer persisted | **No** — `torch` missing; notebook depends on old `archive (2)` layout |
| Transformer | Yes | Yes | Yes | **Imputer/scaler/feature list missing** | **No** — `torch` missing and preprocessing artifacts absent |
| Ensemble | Yes | Yes (saved predictions) | Derived | Frozen config | **Yes** — metrics reproduced exactly from `ensemble_test_predictions.csv` |

**Reproduction check performed:** recomputing AUROC/AUPRC/precision/recall/F1 at threshold 0.81 from `results/ensemble_test_predictions.csv` reproduced `results/ensemble_final_test_comparison.csv` to all printed digits (AUC 0.838431, AUPRC 0.131937, P 0.128814, R 0.349309, F1 0.188219; 234,302 rows; 6,051 patients; 4,194 positive hours). **[V]**

---

## 10. Methodological and statistical issues

### 10.1 Selection and evaluation

- **[V] Triple use of validation.** XGBoost early stopping monitors validation AUC (`eval_set` includes train and validation, `best_iteration=196`). The ensemble weight is then chosen by validation AUC, and the decision threshold by validation F1. Validation is therefore not an unbiased estimate.
- **[V] Threshold heterogeneity.** Saved metrics use threshold 0.5 (XGBoost, GRU, LSTM, Transformer), 0.17 (Random Forest, tuned on validation F1), and 0.81 (ensemble table). Cross-file comparisons at "the reported F1" are invalid.
- **[V] Accuracy is dominated by the negative class.** At 1.8% prevalence, an all-negative classifier achieves 98.2% accuracy. Reported accuracies of 0.91–0.95 (and 0.98 for Random Forest at 0.5) are not evidence of clinical usefulness.
- **[V] No patient-level metrics.** All AUROC/AUPRC are computed over hourly rows. A single long stay contributes up to 336 rows. Patient-level AUROC, the PhysioNet 2019 utility score, and per-patient alarm burden are absent.
- **[V] No calibration.** Raw probabilities are thresholded at 0.81 (far from 0.5) and linearly averaged in the ensemble without calibration. No reliability diagram, Brier score, or calibration slope is reported.
- **[V] No uncertainty.** No confidence intervals, bootstrap, or repeated splits.
- **[V] No lead-time analysis.** The PhysioNet label is defined (by the challenge specification) as positive from 6 hours before onset onward; the notebooks never document or stratify by time-to-onset, so it is impossible to tell whether detections are early or late. **[V]** for the absence of the analysis; the label definition itself is external specification, not repository evidence.
- **[S] Threshold 0.81 is a validation-F1 artefact.** The validation F1 surface is almost flat between 0.80 and 0.86 (0.1575, 0.1587, 0.1580, 0.1585), so 0.81 is not a robust operating point.

### 10.2 Branch asymmetry

- **[V] XGBoost and Random Forest use raw hourly rows with NaNs retained** — no forward fill, no aggregation, no missingness indicators. Apart from `ICULOS` and `HospAdmTime`, these models see almost no temporal history.
- **[V] GRU uses forward fill (causal), training-set medians, standardisation, clipping at ±10, and 34 missingness flags** (74 inputs).
- This is a **design asymmetry**, not leakage: the ensemble combines two models trained on different representations of the same rows. It does mean the ensemble gain cannot be attributed cleanly to architecture.

### 10.3 Sequence-model details

- **[V] GRU inference does not pass a padding mask.** `predict` batches are padded and fed to the GRU without a mask. Because the GRU is unidirectional and outputs at valid positions do not depend on later padding, this does not corrupt predictions; it only wastes compute. Loss and metrics do use the mask.
- **[V] The Transformer had a padding-mask defect that was fixed** in commit `b2c4add`; the corrected results are saved separately as `*_padding_mask.csv`. The older, uncorrected Transformer metrics are not present in `results/`, so there is no stale-metric conflict — but the file naming implies a prior bug.
- **[V] LSTM notebook hygiene.** It contains failed `pip install` cells, `ModuleNotFoundError` cells, a `NameError` on `X_train_pad`, a 10-epoch loop that ran on an 80-patient / 3,078-timestep tensor (test AUC 0.5569), and then a separate full 5-epoch run on all 28,235 patients. The saved LSTM metrics correspond to the full run. Anyone reading the notebook top-to-bottom will encounter misleading intermediate numbers.

### 10.4 Positive-class weighting

`scale_pos_weight` (XGBoost) ≈ 54.54, `balanced_subsample` (Random Forest), `pos_weight` ≈ 54.60 (LSTM), 54.54 (Transformer). All are reasonable for 1.8% prevalence, but none is combined with probability calibration, which is why the F1-optimal thresholds drift far from 0.5.

---

## 11. Clinical score audit

### 11.1 Partial SOFA (`06_sofa.ipynb`) — **not canonical SOFA**

- **[V] Only 4 of 6 components implemented:** coagulation (platelets), liver (total bilirubin), cardiovascular (**MAP only**, no vasopressors), renal (**creatinine only**, no urine output). Respiratory (PaO2/FiO2) and CNS (GCS) are entirely absent.
- **[V] Missing components are left as NaN and excluded from the sum** (`min_count=1`), with `SOFA_Components_Available` recorded. Scores are therefore **not comparable across rows** with different component counts.
- **[V] Two threshold errors against the canonical SOFA table:**
  - Bilirubin: code maps `<= 12.0` to 3 points; canonical is `>= 12.0` → 4.
  - Creatinine: code maps `<= 5.0` to 3 points; canonical is `>= 5.0` → 4.
- **[V] Coverage:** 204,102 validation rows have ≥1 component; only **2,831 rows (≈1.2%)** have all four.
- **[V] Confusion matrices (validation):**
  - Partial SOFA ≥ 2, all rows with ≥1 component — TN 195,068 / FP 5,194 / FN 3,661 / TP 179.
  - 4-component-only subset — TN 1,277 / FP 1,449 / FN 31 / TP 74.
- **[V] The notebook explicitly records "Canonical SOFA reproducible? — No"** and names the unavailable variables.
- **[S] Consequence:** the `Partial_SOFA >= 2` rule is not the Sepsis-3 organ-dysfunction criterion and will disagree with a correct implementation in both directions. It should not be used as a label generator or as a clinical baseline in a new pipeline.

### 11.2 Modified SIRS (`07_sirs.ipynb`) — **not canonical SIRS**

- **[V] Four available criteria:** temperature (>38 or <36 °C), heart rate (>90), respiratory rate (>20) or PaCO2 (<32), WBC (>12 or <4). Band forms are unavailable.
- **[V] Missing criteria are skipped** (`min_count=1`), so `Modified_SIRS` is not comparable across rows.
- **[V] Coverage:** 211,124 validation rows with ≥1 criterion; 5,380 with all four.
- **[V] Confusion matrices (validation):**
  - Modified SIRS ≥ 2, all evaluable rows — TN 174,401 / FP 32,777 / FN 2,783 / TP 1,163.
  - 4-criterion-only subset — TN 3,354 / FP 1,912 / FN 55 / TP 59.
- **[V] The "ROC_AUC" column for individual components is computed on a binary 0/1 score, not a continuous one.** It is arithmetically `(sensitivity + specificity) / 2`, not an ROC AUC. Reporting it as AUC would be misleading.
- **[V] No result artifacts were written** for either SOFA or SIRS; only the two presentation PNGs referenced in code, and those are not present in `results/`.

---

## 12. Privacy, security, and governance

- **[V] No secrets are committed:** `.env` is git-ignored and contains only `PHYSIONET_ROOT`. It was not reproduced in this report.
- **[V] PhysioNet data** is public, de-identified challenge data. `results/patient_summary.csv` contains one row per patient (study ID, stay length, sepsis-hour count, set). This is not PHI, but it is record-level and committed to the repository.
- **[V] MIMIC-IV data** is governed by a PhysioNet credentialed-access Data Use Agreement. It is currently **untracked** (`dataset/` is git-ignored), which is correct. However:
  - 92.7 GB sits inside the working tree; any future archive/zip operation or misconfigured backup could copy it.
  - `results/` artifacts derived from MIMIC must never contain `subject_id`, `hadm_id`, or `stay_id` in committed files.
- **[S] Recommended controls for the new pipeline:** write only aggregate metrics to `results/`, keep row-level outputs in a git-ignored `outputs/` or `data_derived/` directory, and add a CI check that scans committed CSVs for identifier-like columns.

---

## 13. Findings register

Severity: **Critical** = blocks the stated next phase; **High** = materially affects validity or reproducibility; **Medium** = quality/engineering debt; **Low** = hygiene.

| ID | Severity | Status | Finding | Impact | Recommended action (audit only — not performed) |
|---|---|---|---|---|---|
| F-01 | High | **[V]** | Split generator is missing (notebook 03 empty) | Splits cannot be re-derived or audited; blocks defensible MIMIC split design | Implement and commit a deterministic, seeded, patient-level splitter with a manifest |
| F-02 | Critical | **[V]** | No MIMIC-IV code exists at all | The "next dataset" is not started; 92.7 GB is inert | Build a separate `mimic` package; do not extend the PhysioNet notebooks |
| F-03 | High | **[V]** | `torch` absent from `requirements.txt` and `.venv` | Three checkpoints cannot be loaded or re-run | Decide whether to add `torch` (CPU) as an explicit dependency and pin a version |
| F-04 | High | **[V]** | GRU notebook uses macOS absolute paths and missing cached pickles | GRU cannot be reproduced from this machine | Refactor preprocessing into `src/` and re-run once to regenerate artifacts |
| F-05 | Medium | **[V]** | Validation used for early stopping, weight selection, and threshold selection | Validation metrics optimistic; selection overfits validation | Add a three-way split (train / tune / internal-test) or nested CV for the new pipeline |
| F-06 | High | **[V]** | Hour-level metrics only; no patient-level, utility, calibration, or CI | Headline numbers overstate clinical value | Adopt patient-level AUROC/AUPRC + PhysioNet utility + calibration + bootstrap CIs |
| F-07 | Medium | **[V]** | Saved metrics use thresholds 0.5 / 0.17 / 0.81 | Cross-model comparison in `results/` is invalid | Publish one threshold policy and recompute all models at it |
| F-08 | Low | **[V]** | `ensemble_config.json` has `test_evaluated: false` after test evaluation | Stale provenance flag | Make config generation part of the evaluation run |
| F-09 | Medium | **[V]** | LSTM notebook contains broken cells and a misleading 80-patient toy run | High risk of citing wrong numbers | Rewrite as a clean notebook; keep only the full-cohort run |
| F-10 | High | **[V]** | Partial SOFA is non-canonical, with two boundary errors and 2 missing components | Cannot be called SOFA; unusable as a label or baseline | Implement canonical SOFA (6 components, vasopressors, urine output, GCS, PaO2/FiO2) from scratch for MIMIC |
| F-11 | Medium | **[V]** | SIRS component "ROC_AUC" is computed on a binary score | Misleading if reported as AUC | Remove the column or compute AUC on a continuous score |
| F-12 | Medium | **[V]** | SOFA/SIRS results saved only inside notebook outputs | Not machine-readable; cannot be cited reliably | Emit CSV/JSON artifacts with run metadata |
| F-13 | Medium | **[V]** | XGBoost/RF have no temporal features; GRU does | Ensemble gain is confounded by representation | Document the asymmetry or align representations |
| F-14 | Medium | **[V]** | No tests, CI, packaging, README, or `src/__init__.py` | Onboarding and regression risk | Add minimal package structure and smoke tests |
| F-15 | Medium | **[V]** | Transformer imputer/scaler/feature list missing; RF model untracked (1 GB) | Transformer inference not reproducible; RF weights not in git | Persist small preprocessing artifacts in `models/`; store large weights outside git with a manifest |
| F-16 | Medium | **[V]** | MIMIC checksums refer to `.csv.gz`; disk has uncompressed `.csv` | Local data integrity cannot be verified with the provided manifest | Keep original archives or regenerate checksums after a documented extraction |
| F-17 | Low | **[V]** | Double-nested `<table>.csv/<table>.csv` layout | Every loader needs a special case | Handle explicitly in config; do not rename without user approval |
| F-18 | Low | **[V]** | Record-level artifacts committed (`patient_summary.csv`, 14 MB predictions) | Data minimisation / DUA hygiene | Keep record-level outputs git-ignored; commit aggregates only |
| F-19 | Low | **[V]** | `src/config.py` fails on import without `.env`; creates dirs on import | Fragile for CI and fresh clones | Validate env vars explicitly; make directory creation explicit |
| F-20 | Low | **[V]** | Three notebooks share the `08_` prefix; naming is ambiguous | Confusing experiment history | Rename in a future, user-approved cleanup |
| F-21 | Low | **[V]** | GRU inference omits padding mask | Wasted compute only (unidirectional) | Pass the mask for efficiency and consistency |
| F-22 | Medium | **[V]** | No probability calibration despite ensembling raw probabilities | Thresholds are not portable across cohorts (critical for MIMIC) | Add calibration (Platt/isotonic) fitted on a dedicated split |
| F-23 | Medium | **[S]** | MIMIC-IV version not recorded | Reproducibility and itemid stability risk | Record version and checksums in a dataset manifest |
| F-24 | High | **[V]** | No documentation of the 6-hour pre-onset label semantics or onset-time stratification | Cannot distinguish early from late detection | Document label definition and report time-to-onset-stratified performance |

---

## 14. Recommended sequencing (for user decision, not executed)

1. **Freeze the PhysioNet work as-is.** Do not refactor it while adding MIMIC-IV; the risk of breaking published numbers is high.
2. **Fix the two blockers before any MIMIC modelling:** (a) add `torch` to the environment if neural baselines are to be retained, and (b) implement a committed split generator + manifest pattern.
3. **Create a strictly separate MIMIC package** (`src/mimic/`) with its own config, cohort builder, label builder, feature builder, and splitter. It must not import PhysioNet preprocessing.
4. **Implement canonical Sepsis-3** (suspected infection + full SOFA) rather than reusing `Partial_SOFA`.
5. **Adopt a single evaluation contract:** patient-level AUROC/AUPRC, calibration, utility, lead time, bootstrap CIs; one threshold policy.
6. **Then** add sequence models and any twin/agentic layer on top of a validated, reproducible feature store.

---

## 15. Audit limitations

- Notebook outputs were read, not regenerated. A notebook output can be stale relative to its code; this audit treats stored outputs as evidence of what was run, and result CSVs as the authoritative artifacts where both exist.
- Only the ensemble metrics were independently recomputed. XGBoost, Random Forest, GRU, LSTM, and Transformer metrics were cross-checked for internal consistency (confusion matrices vs precision/recall/F1) but not re-derived from raw data.
- MIMIC-IV item-level prevalence, missingness, and cohort counts were **not** computed (that would require scanning tens of GB). The inventory records schema and candidate itemids only.
- No inference was made about clinical validity beyond what the code and reported metrics support.
