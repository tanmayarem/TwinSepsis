# TwinSepsis tool-calling agent (A2)

This is a research prototype. It is not a diagnostic system and it does not
issue treatment recommendations.

## What works

- Loads one real local PhysioNet Challenge 2019 patient by identifier.
- Uses the real, verified `models/xgboost_baseline.json` artifact
  (`PhysioNetXGBRisk`) for the current-hour risk. This is the default backend.
- Optional `--risk-model ensemble` backend: the frozen PhysioNet XGBoost + GRU
  probability blend (`EnsembleRisk`) verified against
  `results/ensemble_test_predictions.csv`.
- Exposes up to seven tools to an LLM: patient state/trend, risk prediction,
  feature attribution, risk trajectory, qSOFA/SIRS, intervention listing and
  scenario projection. Tools whose capability the loaded model lacks are not
  exposed at all.
- Validates every tool argument against JSON Schema and caps a run at eight
  tool calls.
- Assembles the final JSON response from tool outputs. The LLM never supplies a
  probability, a measurement, an attribution or an effect size.
- Writes a JSONL operational trace with tool names, arguments, timestamps,
  status and model identifiers. It does not write patient rows, tool outputs,
  prompts or API keys.
- Runs all tests offline with a mocked LLM.

## Risk model backends

`--risk-model` accepts exactly two values; the default is `xgboost`, so
existing behaviour is unchanged.

| | `xgboost` (default) | `ensemble` |
|---|---|---|
| Class | `PhysioNetXGBRisk` | `EnsembleRisk` |
| Components | XGBoost | XGBoost 0.55 + GRU 0.45 |
| Decision threshold | 0.500 | 0.810 |
| Feature attribution | TreeSHAP (log-odds) | unavailable |
| Intervention simulation | available (effects are `TODO`) | unavailable |
| Interpreter | project `.venv` (Python 3.14) | `C:\TwinSepsis-venvs\gru312` (Python 3.12 + CPU PyTorch) |

The ensemble's weights, decision threshold and component list are read from
`results/ensemble_config.json` (`weighted_probability_ensemble`, XGBoost 0.55,
GRU 0.45, threshold 0.81). They are frozen validation-selected values and
cannot be overridden from the CLI or a caller. Do not retune them at inference
time.

Both components are loaded eagerly. If either XGBoost or the GRU checkpoint
fails to load, the ensemble raises `ModelUnavailableError` and the CLI exits
with code 2 and a JSON error. It never silently falls back to XGBoost-only
risk.

The GRU component reproduces the architecture and preprocessing in
`notebooks/08_gru.ipynb` and `notebooks/09_xgboost_gru_ensemble.ipynb`: the
checkpoint's 40 raw columns, `inf` -> `NaN`, per-patient forward fill, training
medians/means/stds, clipping to +/-10, and 34 observation masks (74 features per
hour). Scoring uses one causal forward pass over the patient's hourly prefix, so
appending future rows never changes past scores. Inference is CPU-only, and
nothing retrains or overwrites `models/xgboost_baseline.json` or
`models/gru_baseline.pt`.

## What is intentionally unavailable

- No MIMIC-IV model or preprocessing exists. Both backends are PhysioNet
  Challenge 2019 models.
- Ensemble-level feature attribution does not exist. XGBoost-only TreeSHAP
  values describe one component, not the blend, so they are not reported as
  ensemble explanations. With `--risk-model ensemble`, `explain_risk` is not
  exposed and the response includes `feature_attribution_unavailable`.
- Intervention simulation is disabled for the ensemble in v1
  (`supports_simulation = False`) because counterfactual re-scoring of the
  blend has not been validated. `simulate_intervention` is likewise not
  exposed, and the response includes `intervention_simulation_unavailable`
  with a matching caveat.
- Intervention effect sizes in `config/assumptions.yaml` are `TODO`. Scenarios
  return `available: false` and null projected risk instead of invented values.
- Both backends score the current ICU hour (PhysioNet hourly label), not a
  multi-hour forecast, and neither supports causal treatment-effect claims.

### Unavailable tools in tools-only responses and traces

`tools_used` and the JSONL trace record every step of the fixed `tools_only`
plan, including steps the loaded model cannot support. Those entries appear
with `ok: false` and `error_type: "tool_unavailable"`; they are rejected
before any model call, carry no patient data, and are **expected output**, not
an execution failure. The LLM path only receives tools the model supports, so
it cannot record a `tool_unavailable` call.

## Provider rate limits and LLM failures

The client never retries an LLM request. Provider errors are reported with
short, safe messages that do not echo the provider's response body. HTTP 429 is
reported as `LLMRateLimitError`; a `Retry-After` header (seconds or HTTP date)
is parsed into `retry_after_s` and reported as metadata, not acted on.

If the LLM fails after one or more tools have already succeeded, the agent keeps
those real results, stops calling the provider, and assembles the normal
structured response from them. That response has
`"summary_source": "deterministic_fallback"`; the summary starts with
"Deterministic system-generated summary (not an LLM response)." and
`limitations` contains `llm_summary_unavailable`. Risk, the frozen-threshold
decision and trend are included only when the completed tools supplied them;
missing values are never filled in. The trace records the successful and failed
tool calls plus an `llm_failure` event and a final `run_end` with
`status: "llm_error_fallback"`.

If no tool succeeded before the failure, the error is re-raised and the CLI
exits with code 2 and a JSON error. The dashboard shows the safe message and
offers its separately labelled deterministic tool sequence.

## Environment

Copy the names from `.env.example` into `.env` and set values there. `.env` is
already Git-ignored.

```text
TWINSEPSIS_LLM_PROVIDER=groq
TWINSEPSIS_LLM_MODEL=<provider-model-id>
TWINSEPSIS_LLM_BASE_URL=https://api.groq.com/openai/v1
TWINSEPSIS_LLM_API_KEY=<secret>
```

`TWINSEPSIS_LLM_BASE_URL` is the OpenAI-compatible root; the client appends
`/chat/completions`. Provider values `groq`, `openai` and
`openai_compatible` are accepted; only the OpenAI-compatible wire protocol is
implemented.

### Ensemble environment

The project `.venv` (CPython 3.14) deliberately does not include PyTorch, so
the GRU/ensemble backend cannot run there. It is installed in a separate
environment instead:

```text
C:\TwinSepsis-venvs\gru312   # CPython 3.12, torch 2.15.0+cpu
```

`requirements-gru.txt` pins that environment's non-torch dependencies. PyTorch
is not published on PyPI under the `+cpu` local version, so install it first
from the official CPU index (see the header of `requirements-gru.txt`). Do not
install torch into `.venv`; the default XGBoost-only path must keep working
without it.

## Example requests

Deterministic demonstration with real patient data and the real model, no LLM
and no API key (the output is labelled `"mode": "tools_only"`):

```powershell
.\.venv\Scripts\python.exe run_agent.py `
  --patient-id p000001 --hour 6 --tools-only
```

The same patient through the LLM agent. This demonstrates state retrieval, risk
prediction, explanation and trajectory in one request:

```powershell
.\.venv\Scripts\python.exe run_agent.py `
  --patient-id p000001 --hour 6 `
  --request "Summarize the current model-estimated risk, the top drivers, and the recent risk trend. If intervention scenarios are unavailable, say so."
```

The ensemble backend must run in the Python 3.12 environment:

```powershell
C:\TwinSepsis-venvs\gru312\Scripts\python.exe run_agent.py `
  --patient-id p000001 --hour 6 --risk-model ensemble --tools-only
```

The ensemble response reports the frozen threshold 0.810 and does not expose
`explain_risk` or `simulate_intervention` to the LLM; `top_drivers` is `null`
and `limitations` contains `feature_attribution_unavailable` and
`intervention_simulation_unavailable`. In `tools_only` mode the fixed
demonstration plan still records those calls, so `tools_used` shows them with
`ok: false` and `error_type: "tool_unavailable"`; they are never executed.

Output is JSON with `risk_summary`, `prediction_source`, `prediction_horizon`,
`top_drivers`, `trajectory`, `scenarios`, `caveats` and
`needs_clinician_review`, plus `clinical_scores`, `limitations`, `tools_used`,
`mode` and `summary_source` (`llm_agent`, `deterministic_fallback` or
`tools_only`).

Set `--dataset-root` if `PHYSIONET_ROOT` is not configured. Use `--no-trace` to
skip trace writing, `--max-tool-calls` to lower the cap, and `--output` to write
the response to a file.

## Tests

XGBoost-only environment (default, no PyTorch). GRU and ensemble tests skip:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Full suite including the GRU checkpoint contract and committed-prediction
regressions:

```powershell
C:\TwinSepsis-venvs\gru312\Scripts\python.exe -m unittest discover -s tests -q
```

Tests use scripted/mocked LLM turns and `httpx.MockTransport`; they make no
network calls.

## Patient privacy

- `SepsisLabel` is dropped when a patient file is loaded.
- The twin contains only the 40 model features for the requested hours.
- The LLM receives tool outputs for one fixed patient and hour, not dataset
  tables.
- Traces omit patient identifiers, raw histories and tool results.
