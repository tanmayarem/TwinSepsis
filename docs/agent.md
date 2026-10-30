# TwinSepsis tool-calling agent (A2)

This is a research prototype. It is not a diagnostic system and it does not
issue treatment recommendations.

## What works

- Loads one real local PhysioNet Challenge 2019 patient by identifier.
- Uses the real, verified `models/xgboost_baseline.json` artifact
  (`PhysioNetXGBRisk`) for the current-hour risk.
- Exposes seven tools to an LLM: patient state/trend, risk prediction, feature
  attribution, risk trajectory, qSOFA/SIRS, intervention listing and scenario
  projection.
- Validates every tool argument against JSON Schema and caps a run at eight
  tool calls.
- Assembles the final JSON response from tool outputs. The LLM never supplies a
  probability, a measurement, an attribution or an effect size.
- Writes a JSONL operational trace with tool names, arguments, timestamps,
  status and model identifiers. It does not write patient rows, tool outputs,
  prompts or API keys.
- Runs all tests offline with a mocked LLM.

## What is intentionally unavailable

- No MIMIC-IV model or preprocessing exists. `EnsembleRisk` loads the frozen
  weights and threshold but raises `NotImplementedError` for prediction.
- PyTorch is not installed, so the PhysioNet GRU checkpoint cannot be loaded.
- Intervention effect sizes in `config/assumptions.yaml` are `TODO`. Scenarios
  return `available: false` and null projected risk instead of invented values.
- The XGBoost model scores the current ICU hour (PhysioNet hourly label), not a
  multi-hour forecast.

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

Output is JSON with `risk_summary`, `prediction_source`, `prediction_horizon`,
`top_drivers`, `trajectory`, `scenarios`, `caveats` and
`needs_clinician_review`, plus `clinical_scores`, `limitations`, `tools_used`
and `mode`.

Set `--dataset-root` if `PHYSIONET_ROOT` is not configured. Use `--no-trace` to
skip trace writing, `--max-tool-calls` to lower the cap, and `--output` to write
the response to a file.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Tests use scripted/mocked LLM turns and `httpx.MockTransport`; they make no
network calls.

## Patient privacy

- `SepsisLabel` is dropped when a patient file is loaded.
- The twin contains only the 40 model features for the requested hours.
- The LLM receives tool outputs for one fixed patient and hour, not dataset
  tables.
- Traces omit patient identifiers, raw histories and tool results.
