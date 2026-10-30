"""Structured JSONL tracing for agent runs.

Traces deliberately contain *operational* facts only: tool names, validated
arguments, timestamps, success/failure status, model identifiers and durations.
They never contain tool outputs, raw clinical histories, patient identifiers,
prompts or API keys.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRACE_DIR = PROJECT_ROOT / "results" / "agent_traces"


class TraceLogger:
    """Append-only JSONL trace writer."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        trace_dir: str | Path | None = None,
        session_id: str | None = None,
        clock: Callable[[], datetime] | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.session_id = session_id or uuid.uuid4().hex
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()

        if not enabled:
            self.path: Path | None = None
            return

        if path is not None:
            resolved = Path(path)
        else:
            directory = Path(trace_dir) if trace_dir is not None else DEFAULT_TRACE_DIR
            stamp = self._clock().strftime("%Y%m%dT%H%M%SZ")
            resolved = directory / f"trace-{stamp}-{self.session_id[:8]}.jsonl"

        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.path = resolved

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        """Append one event and return the payload that was written."""

        payload: dict[str, Any] = {
            "timestamp": self._clock().isoformat(),
            "event": str(event),
            "session_id": self.session_id,
        }
        payload.update(fields)

        if not self.enabled or self.path is None:
            return payload

        line = json.dumps(payload, allow_nan=False, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
        return payload


__all__ = ["TraceLogger", "DEFAULT_TRACE_DIR"]
