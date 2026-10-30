"""Command-line entry point for the TwinSepsis agent.

Examples
--------
Deterministic demonstration with the real PhysioNet XGBoost artifact (no LLM,
no API key).  ``p000001`` is a real local patient; the command does not write
patient rows to disk:

    python run_agent.py --patient-id p000001 --hour 6 --tools-only

Same patient, through the configured LLM provider (requires the four
``TWINSEPSIS_LLM_*`` variables):

    python run_agent.py --patient-id p000001 --hour 6 \
        --request "Summarize the current risk, the top drivers, and the recent trend."

Output is JSON on stdout.  A JSONL operational trace is written under
``results/agent_traces/`` unless ``--no-trace`` is supplied.  Traces contain
tool names, arguments and status only; they never contain patient rows, tool
results, prompts or credentials.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ..patient_source import PatientSourceError, load_physionet_patient
from .agent import AgentContextError, AgentError, SepsisAgent
from .llm import (
    LLMConfigurationError,
    LLMError,
    LLMNetworkError,
    LLMResponseError,
    create_client,
    LLMConfig,
)

DEFAULT_REQUEST = (
    "Summarize the current model-estimated risk, the frozen threshold decision, "
    "the top model drivers, and the recent risk trend for this patient."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python run_agent.py",
        description=(
            "Run the TwinSepsis research-prototype agent for one local "
            "PhysioNet patient and one scored ICU hour."
        ),
    )
    parser.add_argument(
        "--patient-id",
        required=True,
        help="PhysioNet identifier, for example p000001.",
    )
    parser.add_argument(
        "--hour",
        required=True,
        type=int,
        help="1-based ICU hour to score; history is truncated to this hour.",
    )
    parser.add_argument(
        "--request",
        default=DEFAULT_REQUEST,
        help="Natural-language request sent to the LLM (ignored with --tools-only).",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help=(
            "Directory containing training_setA/training_setB. Defaults to "
            "PHYSIONET_ROOT from the environment/.env."
        ),
    )
    parser.add_argument(
        "--tools-only",
        action="store_true",
        help=(
            "Run a fixed, labelled deterministic tool sequence without an LLM. "
            "Still uses the real patient data and real risk model."
        ),
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=8,
        help="Maximum tool calls for this run (1-8, default 8).",
    )
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="Directory for JSONL traces (default: results/agent_traces).",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Do not write a trace file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the JSON response to this path instead of stdout.",
    )
    return parser


def _error_payload(exc: BaseException) -> dict:
    return {
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        twin = load_physionet_patient(
            args.patient_id,
            args.hour,
            dataset_root=args.dataset_root,
        )
    except PatientSourceError as exc:
        print(json.dumps(_error_payload(exc), indent=2), file=sys.stderr)
        return 2

    client = None
    try:
        agent = SepsisAgent(
            twin,
            None,
            max_tool_calls=args.max_tool_calls,
            trace_dir=args.trace_dir,
            trace_enabled=not args.no_trace,
        )
        if args.tools_only:
            response = agent.run_tools_only(args.patient_id, args.hour)
        else:
            config = LLMConfig.from_env()
            client = create_client(config)
            agent = SepsisAgent(
                twin,
                client,
                max_tool_calls=args.max_tool_calls,
                trace_dir=args.trace_dir,
                trace_enabled=not args.no_trace,
            )
            response = agent.run(args.request, args.patient_id, args.hour)
    except (
        LLMConfigurationError,
        LLMNetworkError,
        LLMResponseError,
        LLMError,
        AgentContextError,
        AgentError,
    ) as exc:
        print(json.dumps(_error_payload(exc), indent=2), file=sys.stderr)
        return 2
    finally:
        if client is not None:
            client.close()

    text = json.dumps(response, indent=2, allow_nan=False, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
