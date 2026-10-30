# MIMIC-IV Preprocessing Plan — TwinSepsis

| Field | Value |
|---|---|
| Status | **Proposal for review. Nothing in this document has been implemented or executed.** |
| Date | 2026-10-29 |
| Scope | Design of a new, standalone MIMIC-IV preprocessing pipeline for sepsis prediction |
| Relationship to audit | Companion to `repository_audit.md`, `mimic_table_inventory.csv`, and `existing_experiments.md` in this directory |
| Existing PhysioNet pipeline | **Frozen.** This plan does not modify, reuse, or re-run any PhysioNet notebook, model, split, or result. |
| Code changes in this audit | None. This document and the other three audit files are the only additions. |
| Approval required before Stage 0 | Yes — see §17 (open decisions) |

Legend: **[V]** = verified in the repository during the audit; **[S]** = suspected or inferred; **Decision** = a choice this plan proposes but that requires explicit user sign-off. Candidate itemids are dictionary-verified labels, **not** validated extraction rules.

---

## 1. Purpose and boundary

The repository currently contains a complete PhysioNet Challenge 2019 study and **no MIMIC-IV code at all** **[V]** (finding F-02). The MIMIC-IV data is present but inert. This plan defines how to build a **separate** MIMIC-IV preprocessing pipeline that produces a reproducible, leakage-controlled, hourly feature store and a canonical Sepsis-3 label.

The plan is deliberately limited to **preprocessing and evaluation design**. It stops before model training. Its output should be a set of versioned artifacts that any model — classical, sequence, or later digital-twin layer — can consume without re-implementing cohort or label logic.

### 1.1 Success condition

A reviewer can, from a clean clone plus credentialed MIMIC-IV access:

1. verify the raw-data manifest,
2. rebuild the cohort, labels, features, and splits with one documented command per stage,
3. reproduce the cohort/label counts and the split manifest bit-for-bit,
4. see exactly which rows were excluded and why, and
5. confirm that no feature at prediction time uses information dated after that time.

---

## 2. Explicit non-goals

- **No PhysioNet refactor.** The 11 notebooks, 5 model artifacts, splits, and published metrics stay untouched. The MIMIC package must not import PhysioNet preprocessing.
- **No reuse of `Partial_SOFA` or `Modified_SIRS`.** Both are non-canonical and have documented boundary/coverage defects (F-10, F-11). They are not label generators.
- **No reuse of the PhysioNet 74-input GRU preprocessing.** It is coupled to `.pkl` caches, macOS paths, and a different feature set (F-04).
- **No model training, tuning, or inference in this phase.**
- **No digital-twin, agent/LLM, or UI work.** Those layers are out of scope until the feature store and label are validated.
- **No data download, extraction, re-compression, or renaming.** The double-nested layout is handled in code, not by moving files (F-17).
- **No row-level MIMIC data in git.** No `subject_id`, `hadm_id`, `stay_id`, or record-level timestamps in committed files (F-18).
- **No claims of clinical validity.** This plan produces a research dataset, not a validated clinical tool.

---

## 3. Guiding principles

| # | Principle | Consequence |
|---|---|---|
| P1 | **Separate pipeline, shared conventions** | New `src/mimic/` package; no imports from PhysioNet notebooks; shared only generic utilities if they are first extracted and tested. |
| P2 | **Causality by construction** | Every feature at hour *t* is computed only from observations with `charttime <= t`. No backfill, no future lab, no discharge/death fields. |
| P3 | **Patient-level separation** | Splits are by `subject_id`, never by stay or row. A subject appears in exactly one split. |
| P4 | **One definition, one implementation** | Sepsis-3, SOFA, suspected infection, and the hourly grid each exist in exactly one module with unit tests and a written definition. |
| P5 | **Everything derived is versioned** | Config hash + git commit + input manifest hash + row counts are written next to every output. |
| P6 | **Aggregate-only commits** | Row-level artifacts go to git-ignored `data_derived/`; `results/mimic/` holds aggregates and metrics only. |
| P7 | **Fail loudly** | Schema drift, unexpected itemid units, duplicate keys, or impossible values stop the pipeline; they are not silently coerced. |
| P8 | **Decisions are recorded** | Every ambiguous clinical choice becomes an entry in a decision log with a default, rationale, and owner. |

---

## 4. Pipeline at a glance

```
raw MIMIC-IV (read-only, 31 tables)
        │
  [S0] manifest + checksums + version
        │
  [S1] structural validation (headers, keys, row counts)
        │
  [S2] cohort (adult ICU stays, exclusions, index stay)
        │
  [S3] labels (suspected infection → SOFA → Sepsis-3 onset → prediction target)
        │
  [S4] hourly grid + feature extraction (causal)
        │
  [S5] causal preprocessing (impute, scale, missingness)
        │
  [S6] patient-level splits + manifest
        │
  [S7] evaluation contract + baseline sanity checks
        │
  versioned feature store → (future) model training
```

Each stage writes a machine-readable artifact plus a human-readable summary. A stage may not consume a downstream artifact.

### 4.1 Stage summary

| Stage | Input | Output | Hard gate |
|---|---|---|---|
| S0 Governance | Raw directory | `manifests/mimic_iv_manifest.json` | Every file hashed, version identified |
| S1 Validation | Manifest + dictionaries | `data_derived/mimic/validated/` | 31/31 tables parse; keys unique; expected columns present |
| S2 Cohort | `icustays`, `patients`, `admissions`, `transfers` | `cohort.parquet` + exclusion ledger | 100% of ICU stays accounted for |
| S3 Labels | Cohort + `microbiologyevents`, `prescriptions`/`emar`, `chartevents`, `labevents`, `inputevents`, `outputevents`, `procedureevents` | `labels.parquet` + label QC report | Every sepsis onset traceable to a culture, an antibiotic, and a SOFA event |
| S4 Features | Cohort + labels + event tables | `features_hourly.parquet` | No feature timestamp > prediction timestamp |
| S5 Preprocess | Raw features + split assignment | Imputation/scaling parameters + transformed store | Parameters fitted on train only |
| S6 Splits | Cohort + labels | `splits.parquet` + manifest | Zero subject overlap; counts reconcile |
| S7 Evaluation | Predictions (future) | Metric tables + run card | Test split evaluated once, threshold frozen |

---

## 5. Stage 0 — Governance and data manifest

### 5.1 Access and storage

- MIMIC-IV is credentialed-access data under the PhysioNet DUA. It must remain outside git (`dataset/` is already ignored) **[V]**.
- Derived row-level data goes to a new git-ignored `data_derived/mimic/` tree. Only aggregate outputs go to `results/mimic/`.
- No derived dataset may be copied to cloud storage, shared drives, or notebooks with output committed unless the DUA permits it.
- The pipeline must run offline against the local copy; no download step.

### 5.2 Version identification

The audit could not establish the MIMIC-IV release **[V]**: `SHA256SUMS.txt` references `CHANGELOG.txt` and `LICENSE.txt`, but neither file is present on disk, and only the checksum manifest remains. The schema is consistent with a recent MIMIC-IV release but this is **[S]**.

**Required before S1:**

1. Record the release string from the distribution source (e.g., PhysioNet project version) in the manifest.
2. If the original archive is available, record its hash and size.
3. If it is not available, record `version: "unconfirmed"` explicitly; do not guess.
4. Record the exact file list and the double-nested path layout.

### 5.3 Checksum strategy

The supplied manifest lists `*.csv.gz` while the disk holds uncompressed `*.csv` **[V]** (F-16). Therefore the provided hashes **cannot** verify the local files.

Two supported options (decision D-01):

- **Option A (preferred):** retain or re-acquire the original `.csv.gz` archives outside the working tree, verify the published SHA-256 values, and extract to a read-only staging directory.
- **Option B:** compute a new SHA-256 manifest over the uncompressed CSV files and record it as a *local* manifest, clearly labelled as not the vendor manifest.

Either way, the manifest records: path, bytes, SHA-256, row count (header excluded), column count, and first/last key values (non-identifying aggregates only). Row counts require a full scan of ~93 GB; this is a one-time cost and should be run in the background with progress logging.

### 5.4 Manifest schema (proposed)

```json
{
  "dataset": "mimic-iv",
  "version": "unconfirmed|vX.Y",
  "source": "local|physionet",
  "generated_at_utc": "...",
  "git_commit": "...",
  "files": [
    {"table": "icustays", "module": "icu", "path": "...", "bytes": 0,
     "sha256": "...", "rows": 0, "columns": 0, "format": "csv"}
  ],
  "vendor_manifest": {"present": true, "matches_local_files": false, "reason": "gz vs csv"}
}
```

---

## 6. Stage 1 — Structural validation and ingestion

### 6.1 Path resolver

All 31 tables are stored as `<table>.csv/<table>.csv` **[V]** (F-17). A single resolver in `src/mimic/io.py` must accept a logical table name and return the real path; no notebook may hard-code the nesting.

### 6.2 Validation checks

| Check | Rule | Failure mode |
|---|---|---|
| Table presence | All 31 inventoried tables exist | Hard stop |
| Header equality | Disk header equals the inventory header exactly | Hard stop; schema drift |
| Key uniqueness | `icustays.stay_id`, `admissions.hadm_id`, `patients.subject_id`, `chartevents` has no natural key (do not assume) | Hard stop on dictionary keys |
| Referential integrity | Every `stay_id` maps to one `hadm_id` and one `subject_id`; every `hadm_id` maps to one `subject_id` | Hard stop |
| Time ordering | `intime < outtime`; `admittime <= intime`; `dischtime >= outtime` (allow documented exceptions, log them) | Quarantine + report |
| Null policy | Nulls allowed only in documented columns; `valuenum` may be null when `value` is text | Report counts |
| Units | Unit strings for selected itemids are within an allow-list per feature | Quarantine + report |
| Duplicates | Duplicate `(stay_id, charttime, itemid)` rows counted, not silently dropped | Report; dedup rule in §9.4 |

### 6.3 Reading strategy

- `chartevents` (~40 GB), `labevents` (~17.5 GB), `emar_detail` (~8.3 GB), `emar` (~6 GB), `poe` (~4.9 GB), and `pharmacy` (~3.8 GB) must be read in chunks with explicit `usecols` and `dtype` maps.
- Use the already-installed `pyarrow`/`pandas` stack; do not add a new query engine without approval.
- Filter early by **cohort keys** and **itemid allow-list**. Never load a full event table into memory.
- Write intermediate partitions by table and, where useful, by `stay_id` bucket, so stages are restartable.
- Log rows read, rows retained, and rows rejected per filter. These counts are part of the QC report.

### 6.4 Dictionary snapshots

`d_items` (4,095 rows) and `d_labitems` (1,650 rows) are small **[V]**. Snapshot the selected rows (itemid, label, abbreviation, category, unitname, linksto/fluid) into the derived store so a future dictionary change is detectable.

---

## 7. Stage 2 — Cohort construction

### 7.1 Base population

Start from `icustays` (94,458 stays, 65,366 subjects, 85,242 admissions with ICU) **[V]**. The ICU stay, not the hospital admission, is the analysis unit because charting, ventilation, vasopressors, and urine output are ICU-scoped.

### 7.2 Proposed inclusion criteria

| # | Criterion | Proposed default | Notes |
|---|---|---|---|
| C1 | Adult at ICU admission | age ≥ 18 | Age reconstructed as `anchor_age + (year(intime) − anchor_year)`; cap at 91+ per MIMIC convention **[S]** |
| C2 | Valid ICU interval | `intime < outtime`, LOS > 0 | Log invalid intervals |
| C3 | Minimum observation | ICU LOS ≥ 6 h | Ensures at least one prediction opportunity; **decision D-02** |
| C4 | Index stay | First ICU stay per `subject_id` | Primary analysis; all-stay sensitivity analysis **decision D-03** |
| C5 | Linkage | `subject_id`/`hadm_id` present in `patients`/`admissions` | Drop and report unlinked rows |
| C6 | Pre-existing sepsis | Not excluded by default | Excluding prevalent sepsis changes the prediction question; **decision D-04** |

### 7.3 Exclusion ledger

Every excluded stay gets a row in an aggregate ledger with reason code and count:

```text
reason_code, n_stays, n_subjects, pct_of_base
missing_linkage, ...
los_below_minimum, ...
age_unknown, ...
not_first_stay, ...
```

The ledger must reconcile exactly: `base − exclusions = analytic cohort`. No exclusion may be applied only in a notebook.

### 7.4 Cohort output contract

`cohort.parquet`, one row per analytic ICU stay:

| Column | Type | Description |
|---|---|---|
| `subject_id` | int64 | Subject key (never committed) |
| `hadm_id` | int64 | Admission key |
| `stay_id` | int64 | ICU stay key |
| `intime`, `outtime` | timestamp | ICU interval |
| `los_hours` | float | `(outtime − intime)` in hours |
| `age_at_intime` | float | Reconstructed age |
| `gender` | category | From `patients` |
| `first_careunit` | category | From `icustays` |
| `admission_type` | category | From `admissions` |
| `anchor_year_group` | category | For temporal sensitivity analysis |
| `index_stay` | bool | First stay per subject |
| `exclusion_reason` | category | Null for included stays |

---

## 8. Stage 3 — Label construction

This is the highest-risk stage. The label must be **canonical Sepsis-3**, not the repository's partial scores. The implementation should follow the published MIT-LCP `sepsis3` concept logic, re-derived in this repository and validated against its own invariants. External code may be consulted for semantics, but the committed implementation must be readable and tested.

### 8.1 Suspected infection (SI)

Proposed definition (decision D-05):

- **Antibiotic evidence:** a qualifying antimicrobial administration or order.
- **Culture evidence:** a microbiology specimen collection.
- **Pairing window:** antibiotic within 72 h after culture **or** culture within 24 h after antibiotic.
- **SI time:** the earlier of the antibiotic time and the culture time.
- **Source preference:** use `emar` administration timestamps where available (actual given events), falling back to `prescriptions.starttime` (order start) and cross-checking `pharmacy`. Never use `storetime` as the clinical time.
- **Antibiotic list:** a versioned, human-reviewable list of drug names/classes. Free-text `drug` and `medication` fields require normalization; the list and its version are part of the artifact.

**Why this matters:** order time is not administration time. Using orders alone can move SI earlier and inflate the apparent prediction window. The fallback must be recorded per row (`si_time_source`) and sensitivity-analysed.

### 8.2 SOFA

Implement all six components for the first time in this repository (F-10):

| Component | Proposed inputs | Key cautions |
|---|---|---|
| Respiratory | PaO2 (ABG), FiO2, PEEP, ventilation status | PaO2/FiO2 ratio; assume FiO2 0.21 when undocumented only under a stated rule; ventilation status from `procedureevents`/chartevents |
| Coagulation | Platelet count | Use nadir in window |
| Liver | Total bilirubin | Use maximum in window; correct boundaries |
| Cardiovascular | MAP; vasopressor agent and rate | MAP-only vs vasopressor tier; normalize rates to µg/kg/min; document units |
| CNS | GCS eye/verbal/motor | Sum components; handle sedated/intubated patients with a stated rule; do not silently impute 15 |
| Renal | Creatinine; urine output | Use worst creatinine or 24-h urine output; document oliguria window |

**Computation rule:** for each candidate SI time, compute the worst SOFA value in the window `[SI − 48 h, SI + 24 h]`. A stay is labelled septic if worst SOFA ≥ 2 and an SI event exists. Sepsis onset is the earlier of the SI time and the time the SOFA threshold is first met.

**Baseline SOFA:** the primary analysis assumes baseline 0 when no pre-ICU data exists; a sensitivity analysis computes a pre-ICU baseline where available. This assumption must be stated in every result (decision D-06).

**Validation gate (hard):**

1. Every component value maps to exactly one canonical SOFA point value.
2. Boundary tests cover the published cut-points (including the two boundaries the old notebook got wrong).
3. Unit tests cover missing components (score contribution must be explicit, not silently zero-filled into a comparable total).
4. A sample of stays is manually traceable from raw event → component score → total → onset.
5. Component coverage is reported per stay; low coverage is a finding, not something to hide by imputation.

### 8.3 Onset and prediction target

Two different objects must not be conflated:

- **Label onset** `t_sepsis`: the clinical Sepsis-3 onset time defined above.
- **Prediction target** at hour *t*: `y(t) = 1` if `t_sepsis ∈ (t, t + H]`, where `H` is the prediction horizon (proposed **6 h**, decision D-07). Rows at or after onset are not negative controls; they are excluded from the prediction task or handled as a separate "already septic" stratum.

Also record:

- `hours_to_onset = t_sepsis − t`
- `sepsis_before_icu` (SI/SOFA met before `intime`)
- `onset_in_stay` (onset during the ICU interval)
- `censoring_reason` (discharge, death, end of stay before horizon)

**Leakage guard:** the onset time and any label-derived field must never be used as a feature, and feature extraction must not read the label table.

### 8.4 Label QC report (aggregate only)

Produce `results/mimic/label_qc.json` containing:

- base stays, included stays, septic stays, sepsis prevalence
- SI events per stay, antibiotic source distribution (`emar` vs `prescriptions`), culture source distribution
- SOFA component coverage (fraction of stays with each component)
- onset location: before ICU / in first 24 h / later
- missingness of each SOFA input
- counts of stays with SOFA ≥ 2 but no SI, and SI but SOFA < 2
- comparison against the reference concept on the same cohort, if a reference extract is available

No prevalence target is imposed. If the computed prevalence is implausible, the response is to audit the definition, not to tune it to a number.

---

## 9. Stage 4 — Hourly grid and feature extraction

### 9.1 Time base

- Hour 0 = `intime`. Hour *h* covers `[intime + h·1h, intime + (h+1)·1h)`.
- Rows exist for `h = 0 … floor(los_hours) − 1`; the final partial hour is dropped and counted.
- Pre-ICU data (ED, ward) may be used only in a clearly separated baseline window and only if timestamped before `intime`. It is **not** placed into negative-hour feature rows by default (decision D-08).
- All timestamps are absolute; shifted calendar dates are never interpreted as real dates. `anchor_year_group` is used only for temporal split sensitivity.

### 9.2 Feature families

| Family | Examples | Source | Cadence |
|---|---|---|---|
| Demographics / context | age, gender, admission type, first care unit, hours since ICU admission | `patients`, `admissions`, `icustays` | static |
| Vital signs | HR, RR, SBP, DBP, MAP, SpO2, temperature, EtCO2 | `chartevents` | hourly |
| Neurological | GCS eye, verbal, motor, total | `chartevents` | hourly / as charted |
| Respiratory support | FiO2, PEEP, tidal volume, ventilator mode, invasive ventilation flag | `chartevents`, `procedureevents` | hourly |
| Urine output | hourly total, cumulative 24 h | `outputevents` | hourly |
| Laboratory | CBC, BMP, LFTs, coagulation, ABG, lactate | `labevents` | as resulted; carry forward with limit |
| Vasoactive therapy | agent, rate, rate unit, normalized µg/kg/min | `inputevents` | hourly infusion state |
| Fluids | crystalloid/colloid volume in prior hour and cumulative | `inputevents` | hourly |
| Derived severity | causal SOFA components, shock index, SIRS-like count (feature only, clearly named) | derived | hourly |
| Missingness | per-feature "observed in lookback" indicators | derived | hourly |

### 9.3 Candidate itemids (dictionary-verified, extraction-unvalidated)

These were confirmed to exist with the stated labels in `d_items`/`d_labitems` **[V]**. Their clinical suitability, unit consistency, and completeness are **[S]** until Stage 1/4 QC passes. This is not an exhaustive list.

| Role | Candidate itemids | Notes |
|---|---|---|
| Heart rate | 220045 | chartevents |
| Respiratory rate | 220210 | chartevents; also check 224690 |
| Temperature | 223761 (F), 223762 (C) | convert to °C |
| SpO2 | 220277 | chartevents |
| Arterial BP | 220050 (SBP), 220051 (DBP), 220052 (MAP) | chartevents |
| Non-invasive BP | 220179 (SBP), 220180 (DBP), 220181 (MAP) | chartevents |
| GCS | 220739 (eye), 223900 (verbal), 223901 (motor) | text and numeric variants exist |
| FiO2 | 223835 | chartevents; percent vs fraction must be normalized |
| PEEP | 220339 | set value; check for measured variant |
| Ventilation | 225792 (invasive), 225794 (non-invasive), 226260 (mode) | confirm `linksto` and table |
| Admission weight | 226512 | chartevents |
| Daily weight | 224639 | chartevents |
| Urine output | 226559, 226560, 226627, 226631, 226713, 227489 | outputevents; exclude non-urine drains |
| Norepinephrine | 221906 | inputevents; rate unit critically important |
| Epinephrine | 221289 | inputevents |
| Dopamine | 221662 | inputevents |
| Dobutamine | 221653 | inputevents |
| Phenylephrine | 221749 | inputevents |
| Vasopressin | 222315 | inputevents; units/min, not µg/kg/min |
| Creatinine | 50912 | labevents |
| Total bilirubin | 50885 | labevents |
| Platelets | 51265 | labevents |
| WBC | 51301, 51755 | labevents |
| Lactate | 50813, 52442 | labevents |
| PaO2 | 50821 | labevents (ABG) |
| PaCO2 | 50818 | labevents (ABG) |
| pH | 50820 | labevents (ABG) |
| Bicarbonate | 50882 | labevents |
| Sodium | 50983 | labevents |
| Potassium | 50971 | labevents |
| Chloride | 50902 | labevents |
| BUN | 51006 | labevents |
| Haemoglobin | 51222 | labevents |
| Haematocrit | 51221 | labevents |
| INR | 51237 | labevents |
| PTT | 51275, 52923 | labevents |
| EtCO2 | 228640 | chartevents |

**Rule:** no itemid enters the feature store until its unit distribution, null rate, and value range are reported for the cohort. A single itemid may map to multiple units (e.g., temperature, weight, FiO2); conversion is explicit and tested.

### 9.4 Aggregation rules

- **Vitals:** last non-null `valuenum` in the hour. If multiple at the same `charttime`, take the mean of numeric values and log the duplication. Do not use `storetime` to choose.
- **Labs:** most recent result with `charttime <= t` within a lookback (proposed 24 h for chemistry, 6 h for blood gas). No result after *t*.
- **GCS:** sum components only when all three are present in the same lookback; otherwise emit components separately and a total of null. Do not assume 15 for sedated patients.
- **Urine output:** sum `value` in the hour; convert mL to mL/kg/h using the most recent weight. Negative or impossible values are quarantined.
- **Vasopressors:** infusion state at the end of the hour; if rate changes within the hour, use the maximum rate and record the change. Normalize to µg/kg/min for catecholamines; keep vasopressin in units/min and flag it.
- **Ventilation:** carry the most recent documented status forward within the stay; do not infer ventilation from a single PEEP value without a stated rule.
- **Missingness:** for each feature, emit `observed_<feature>` = 1 if any valid value existed in the feature's lookback at *t*. These are features, not post-hoc diagnostics.

### 9.5 Leakage controls (must be enforced in code)

| Risk | Control |
|---|---|
| Future lab/vital | Hard filter `charttime <= t`; assert in tests |
| `storetime` used as clinical time | `storetime` never enters feature values; optional sensitivity analysis only |
| Discharge/death fields | `dischtime`, `deathtime`, `hospital_expire_flag`, `dod` excluded from features |
| Label leakage | Label table not joined until after feature extraction; no `t_sepsis`-derived feature |
| Post-onset rows | Prediction rows stop at onset; no "negative" rows after onset |
| Same-subject leakage | Subject-level splits only |
| Pre-ICU future data | Pre-ICU events included only if `charttime < intime` and only in the baseline window |
| Itemid drift | Dictionary snapshot + allow-list; unknown itemid → quarantine |
| Unit drift | Unit allow-list per itemid; conversion tested |

**Automated leakage test:** for a random sample of stays and hours, recompute features from a truncated event table containing only rows with `charttime <= t` and assert equality with the full-table result. This is the single most important test in the pipeline.

---

## 10. Stage 5 — Causal preprocessing

### 10.1 Fit/apply separation

All data-dependent transformations are fitted on the **training split only** and then applied unchanged to tune and test.

| Transform | Proposed rule | Notes |
|---|---|---|
| Forward fill | Within stay, max 6 h for vitals and 24 h for labs | Never across ICU boundaries; record fill age |
| Median imputation | Train medians per feature | Missingness indicators preserve information |
| Scaling | Robust scaler (median/IQR) or standardizer | Fit on train; clip at ±10 |
| Unit conversion | Explicit per itemid | Temperature, weight, FiO2, vasopressor rate |
| Categorical encoding | One-hot for low-cardinality; ordinal only if ordered | Fit categories on train; unseen → "other" |
| Feature selection | None before the first locked evaluation | Avoid selection leakage |

### 10.2 Missingness as signal

Missingness is clinically informative in MIMIC (a lab not ordered is not a random event). Emit per-feature observation flags plus `hours_since_last_observation` for key labs. Do not use a single global "missingness score" that hides which variable is missing.

### 10.3 Persisted preprocessing artifacts

Persist a small, versioned JSON/parquet bundle in `models/mimic/preprocess/`:

- feature list and order
- train medians / IQR / means / SDs
- category vocabularies
- unit conversion table
- clip bounds
- config hash and git commit

This directly addresses F-15 (the Transformer's missing imputer/scaler/feature list) and prevents a repeat.

---

## 11. Stage 6 — Splits

### 11.1 Primary split

- Unit: `subject_id`.
- Proportions: **70% train / 15% tune / 15% test** (decision D-09).
- Stratification: sepsis status (from the final label) and `anchor_year_group` to balance era.
- Seed: configurable, default `42`; the seed is recorded.
- Every subject appears in exactly one split; every stay inherits its subject's split.
- The test split is **locked** after generation: its IDs are written once, hashed, and not used for any choice until final evaluation.

### 11.2 Split manifest

```json
{
  "seed": 42,
  "unit": "subject_id",
  "proportions": {"train": 0.70, "tune": 0.15, "test": 0.15},
  "counts": {"train": {"subjects": 0, "stays": 0, "septic": 0},
             "tune":  {"subjects": 0, "stays": 0, "septic": 0},
             "test":  {"subjects": 0, "stays": 0, "septic": 0}},
  "overlap": {"train_tune": 0, "train_test": 0, "tune_test": 0},
  "stratification": ["sepsis_status", "anchor_year_group"],
  "sha256_of_id_lists": "..."
}
```

### 11.3 Temporal sensitivity split

Because dates are shifted but `anchor_year_group` preserves order **[V]**, also generate a **temporal** split (train on earlier groups, test on later groups) as a secondary analysis. Report both random and temporal results. A model that only works on a random split is not robust to era shift.

### 11.4 Split tests

- zero overlap between any pair
- union equals the analytic cohort's subject set
- per-split sepsis prevalence and era distribution reported
- counts reconcile with the cohort ledger
- re-running the splitter with the same seed and config reproduces the same ID lists exactly

---

## 12. Stage 7 — Evaluation contract

This section defines what future modelling must report. It is part of the preprocessing contract because the feature store must support these metrics.

### 12.1 Two levels, never mixed

| Level | Definition | Purpose |
|---|---|---|
| **Hour-level** | One row per eligible hour; label as defined in §8.3 | Comparable to PhysioNet-style reporting; operational alarm analysis |
| **Patient-level** | One row per stay; score = max risk in the eligible window; label = any onset during stay | Clinical screening performance; avoids long-stay dominance |

Both are reported. A single headline number is not acceptable. The old pipeline reported hour-level only (F-06).

### 12.2 Metrics

- AUROC and AUPRC at both levels, with **patient-level bootstrap 95% CIs** (1,000 resamples by `subject_id`).
- Sensitivity, specificity, PPV, NPV, F1 at the **pre-specified operating threshold**.
- Calibration: reliability curve, Brier score, calibration slope/intercept.
- Utility: PhysioNet 2019 utility (TP − FN − 0.05·FP, normalized) adapted to the MIMIC cohort; state the adaptation explicitly.
- Lead time: distribution of hours between first alarm and onset; performance stratified by time-to-onset.
- Alarm burden: alarms per patient-day at the operating threshold.
- Subgroups: age band, sex, race, first care unit, admission type, `anchor_year_group`; report counts and CIs, not just point estimates.

### 12.3 Threshold policy

One threshold, chosen on the **tune** split by a pre-registered rule (e.g., maximize utility subject to alarms ≤ a stated budget), then frozen and applied once to test (F-07). Report the threshold, the selection rule, and the tune-set curve alongside test metrics.

### 12.4 Test-set discipline

- No test-set inspection during feature or model development.
- One final evaluation per model family; the run card records the git commit and artifact hashes.
- Any re-run after a code change is a new run with a new run ID; old results are retained, not overwritten.
- `test_evaluated` must be written by the evaluation job itself, not a stale config (F-08).

### 12.5 Sanity baselines

Before any model is credited, report:

- prevalence-only baseline (predict the base rate)
- an all-negative classifier's accuracy (to expose the class-imbalance trap)
- a simple early-warning score (e.g., qSOFA or a small logistic model) as a reference
- the old `Partial_SOFA` rule is **not** an acceptable baseline; it is non-canonical

---

## 13. Artifact and directory contract

Proposed layout (nothing created in this audit):

```text
src/mimic/
  __init__.py
  config.py          # env-driven; no import side effects
  io.py              # nested-path resolver, chunked readers
  manifest.py        # hashing + version + row counts
  validate.py        # schema/key/unit checks
  cohort.py
  labels/
    __init__.py
    suspected_infection.py
    sofa.py
    sepsis3.py
  features.py
  preprocess.py
  splits.py
  evaluate.py
  cli.py
configs/mimic/
  cohort.yaml
  labels.yaml
  features.yaml
  preprocess.yaml
  splits.yaml
manifests/
  mimic_iv_manifest.json
  split_manifest.json
docs/mimic/
  definitions.md     # Sepsis-3, SOFA, SI, prediction target
  decisions.md       # decision log
  data_dictionary.md # selected itemids + units
data_derived/mimic/  # git-ignored, row-level
  validated/
  cohort.parquet
  labels.parquet
  features_hourly.parquet
  splits.parquet
results/mimic/       # committed aggregates only
  cohort_summary.json
  label_qc.json
  split_summary.json
  evaluation/
models/mimic/
  preprocess/        # small JSON/parquet artifacts
  <model>/           # weights outside git if large
tests/
  test_io.py
  test_cohort.py
  test_sofa_boundaries.py
  test_suspected_infection.py
  test_leakage.py
  test_splits.py
```

### 13.1 Artifact registry

Every derived artifact records:

```json
{
  "run_id": "mimic-<stage>-<timestamp>-<shorthash>",
  "stage": "labels",
  "git_commit": "...",
  "config_sha256": "...",
  "input_manifest_sha256": "...",
  "rows_in": 0,
  "rows_out": 0,
  "rows_rejected": 0,
  "created_at_utc": "...",
  "software": {"python": "...", "pandas": "...", "pyarrow": "..."}
}
```

### 13.2 Commit policy

- Commit: code, configs, tests, docs, manifests, aggregate QC/metrics.
- Do not commit: row-level parquet/CSV, model weights above the repo's size policy, raw MIMIC files, `.env`.
- Add a pre-commit check that rejects CSVs containing `subject_id`, `hadm_id`, `stay_id`, `charttime`, `intime`, `outtime`.
- Keep the PhysioNet `results/` untouched.

---

## 14. Reproducibility requirements

1. **One command per stage**, e.g. `python -m mimic.cli cohort --config configs/mimic/cohort.yaml`.
2. **Deterministic** given seed, config, and input manifest. Sort keys explicitly; never rely on file order or hash iteration order.
3. **Idempotent**: re-running a stage writes to a new run directory or verifies the existing hash; it does not silently append.
4. **Restartable**: long scans checkpoint by table and partition.
5. **Tested**: unit tests for definitions and an integration smoke test on a small, de-identified fixture (not committed raw data).
6. **Documented**: `docs/mimic/definitions.md` is the single source of truth; code comments reference it.
7. **Pinned**: add only the dependencies actually used; if `torch` is needed for later models, pin it explicitly (F-03). This plan does not change `requirements.txt`.

---

## 15. Validation gates

| Gate | When | Pass criterion | On failure |
|---|---|---|---|
| G0 Manifest | Before any processing | Every file hashed; version state recorded; vendor/local mismatch explained | Stop |
| G1 Schema | After S1 | 31/31 tables parse; headers match; keys valid; referential integrity holds | Stop; reconcile before proceeding |
| G2 Cohort | After S2 | Ledger reconciles 100%; no duplicate `stay_id`; age/LOS ranges plausible | Stop; investigate exclusions |
| G3 Label | After S3 | Boundary tests pass; every onset traceable; component coverage reported; SI source distribution reported | Stop; do not model |
| G4 Leakage | After S4 | Truncated-table recomputation test passes on all sampled rows; no forbidden columns in feature store | Stop; fix extractor |
| G5 Preprocess | After S5 | Train-only fit verified; no NaN in model-ready matrix; flags consistent | Stop |
| G6 Splits | After S6 | Zero subject overlap; union reconciles; seed reproduces | Stop |
| G7 Evaluation | Before any result is cited | Threshold frozen; CIs computed; test run card complete; aggregate-only outputs | Do not report |

Each gate emits a signed (hashed) report. A gate cannot be waived silently; waivers are recorded in `docs/mimic/decisions.md`.

---

## 16. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Ambiguous SI definition shifts onset | High | High | Versioned antibiotic list; `si_time_source` per row; sensitivity analysis |
| SOFA component coverage is low | High | High | Report coverage; avoid silent zero-fill; restrict to stays with sufficient data or use explicit missing-data handling |
| Vasopressor unit errors | Medium | High | Unit allow-list; per-agent normalization; spot-check raw rows |
| FiO2 documented as 0.21 vs 21 vs 100 | High | Medium | Explicit conversion; range validation; missing when ambiguous |
| GCS in sedated patients | High | High | Do not assume 15; record sedation; sensitivity analysis |
| chartevents size causes OOM | High | Medium | Chunked reads, itemid filter, column projection |
| Itemid/version drift | Medium | High | Dictionary snapshot + manifest + version record |
| Label leakage through `storetime` | Medium | High | `charttime` only; optional storetime sensitivity analysis |
| Test-set contamination by repeated looks | Medium | High | Locked split + one-shot evaluation + run cards |
| Row-level data committed | Low | Critical (DUA) | Git-ignore + pre-commit identifier scan + aggregate-only results |
| Scope creep into PhysioNet code | Medium | High | Separate package; no imports; PhysioNet frozen |
| Compute/time underestimation | High | Medium | Stage checkpoints; sample-mode for development; full run scheduled explicitly |

---

## 17. Open decisions requiring sign-off

Each decision needs a yes/no or a chosen option before implementation. Defaults are proposals, not commitments.

| ID | Decision | Proposed default | Why it matters |
|---|---|---|---|
| D-01 | Checksum strategy: re-acquire `.csv.gz` vs hash local `.csv` | Re-acquire if possible; else local manifest | Determines whether integrity is vendor-verifiable (F-16) |
| D-02 | Minimum ICU LOS | 6 h | Trades cohort size against prediction opportunities |
| D-03 | Index stay rule | First ICU stay per subject (primary); all stays (sensitivity) | Independence vs sample size |
| D-04 | Exclude pre-existing sepsis | No (primary); yes (sensitivity) | Changes the clinical question |
| D-05 | SI pairing window and antibiotic source | 72 h after culture / 24 h after antibiotic; `emar` first | Largest single driver of label prevalence |
| D-06 | Baseline SOFA | Assume 0 when no pre-ICU data; sensitivity with computed baseline | Affects who is labelled septic |
| D-07 | Prediction horizon | 6 h | Comparability with PhysioNet and clinical actionability |
| D-08 | Pre-ICU data in features | Baseline window only; not negative-hour rows | Leakage and realism |
| D-09 | Split proportions | 70/15/15, seed 42, stratified by sepsis + era | Reproducibility and tuning budget |
| D-10 | Primary metric | Patient-level AUROC + AUPRC; hourly secondary | Prevents long-stay dominance (F-06) |
| D-11 | Threshold selection rule | Maximize utility subject to alarm budget, on tune only | Prevents test-set tuning (F-07) |
| D-12 | Utility definition | PhysioNet 2019 utility, adapted and documented | Cross-study comparability |
| D-13 | Publication naming | Never call the old scores SOFA/SIRS; reserve those names for the canonical implementation | Scientific accuracy (F-10, F-11) |
| D-14 | Digital-twin scope | Defer; define only after the feature store is validated | Avoid building on an unvalidated base |
| D-15 | Whether to retain neural baselines | Defer; requires an explicit `torch` dependency decision | F-03 |

---

## 18. Traceability to audit findings

| Finding | How this plan addresses it |
|---|---|
| F-01 (split generator missing) | S6 commits a seeded, patient-level splitter with a manifest and reproduction test |
| F-02 (no MIMIC code) | New `src/mimic/` package, explicitly separate from PhysioNet |
| F-03 (`torch` missing) | Out of scope here; flagged as D-15 before any neural work |
| F-04 (GRU macOS paths / missing caches) | No reuse of GRU preprocessing; all artifacts regenerated in-repo |
| F-05 (validation reuse) | Three-way train/tune/test; test locked; threshold and weights chosen on tune only |
| F-06 (hour-level only) | Two-level evaluation contract with patient-level primary metrics, CIs, calibration, utility, lead time |
| F-07 (threshold heterogeneity) | One pre-specified threshold policy |
| F-08 (stale `test_evaluated`) | Run card written by the evaluation job itself |
| F-09 (LSTM notebook mess) | Clean module + CLI; no notebook-only logic |
| F-10 (partial SOFA) | Canonical six-component SOFA with boundary tests and coverage reporting |
| F-11 (binary "ROC_AUC") | No reuse; metrics computed on continuous scores |
| F-12 (results only in notebook outputs) | Machine-readable artifacts at every stage |
| F-13 (representation asymmetry) | Single documented feature contract; any model consumes the same store |
| F-14 (no tests/CI/package) | `src/mimic/` package, tests, and gates; CI proposal included |
| F-15 (missing preprocessing artifacts) | Persisted imputer/scaler/feature bundle in `models/mimic/preprocess/` |
| F-16 (checksum mismatch) | Manifest strategy with vendor/local distinction (D-01) |
| F-17 (nested paths) | Single path resolver |
| F-18 (record-level artifacts committed) | Aggregate-only `results/`; identifier scan in pre-commit |
| F-19 (config import fragility) | Env validation without import side effects |
| F-20 (notebook naming) | No new notebooks required; modules + CLI |
| F-21 (GRU mask) | Not applicable to preprocessing; noted for future modelling |
| F-22 (no calibration) | Calibration is a required evaluation output |
| F-23 (MIMIC version unrecorded) | Version field in manifest; `unconfirmed` allowed but explicit |
| F-24 (label semantics undocumented) | `docs/mimic/definitions.md` defines SI, SOFA, onset, and prediction horizon |

---

## 19. Definition of done for this plan

The preprocessing phase is complete only when all of the following exist and pass:

1. A verified data manifest with an explicit version state.
2. A cohort ledger that reconciles exactly.
3. A canonical Sepsis-3 implementation with boundary tests and traceable onsets.
4. An hourly feature store with a passing truncated-table leakage test.
5. Train-only preprocessing artifacts persisted and hashed.
6. A reproducible patient-level split with zero overlap and a locked test set.
7. Aggregate QC reports for cohort, labels, features, and splits.
8. A written decision log covering D-01…D-15.
9. A documented evaluation contract ready for model runs.
10. No modification to the PhysioNet pipeline and no row-level MIMIC data in git.

Until then, no MIMIC model result should be reported.

---

*End of plan. No implementation was performed.*
