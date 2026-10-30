"""Streamlit dashboard for the TwinSepsis research prototype.

Run it with::

    C:\\TwinSepsis-venvs\\gru312\\Scripts\\python.exe -m streamlit run dashboard.py

The page is a thin renderer.  Patient loading, model loading, scoring,
trajectory, attribution, clinical scores and scenarios all come from
:mod:`twinsepsis.dashboard.service`, which delegates to the existing agent
backend.  Nothing clinical is computed here.

Honesty rules enforced by the layout:

* every risk number is labelled a research-model estimate, not a diagnosis;
* the historical chart is explicitly *not* a forecast and never extrapolates;
* feature attribution is only shown for models that support it (never the
  ensemble, and never XGBoost SHAP relabelled as ensemble attribution);
* intervention scenarios are only shown for models with simulation enabled, and
  are labelled model-based scenarios rather than causal treatment effects;
* missing measurements are reported as missing, never imputed or hidden.
"""

from __future__ import annotations

import html
import json
import math
import platform
from typing import Any, Sequence

import pandas as pd
import streamlit as st

from ..agent.agent import AgentError
from ..agent.llm import LLMConfig, LLMError, create_client
from ..patient_source import PATIENT_ID_PATTERN
from . import charts, theme
from .service import (
    MODEL_CHOICES,
    MODEL_LABELS,
    MODEL_SUMMARIES,
    DashboardError,
    DashboardService,
    llm_status,
    trajectory_window_options,
)

EXAMPLE_PROMPTS: tuple[tuple[str, str], ...] = (
    (
        "Current risk",
        "Summarize the current model-estimated risk and the frozen threshold decision.",
    ),
    (
        "Recent trend",
        "Explain the recent risk trend and how many hours it covers.",
    ),
    (
        "Abnormal vitals",
        "What are the most recent abnormal vital measurements?",
    ),
    (
        "Scores and gaps",
        "Which clinical scores are available, and what information is missing?",
    ),
)


# ---------------------------------------------------------------------------
# Cached backend access
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_service() -> DashboardService:
    """One service (and therefore one model cache) per server process."""

    return DashboardService()


@st.cache_data(show_spinner=False, max_entries=8)
def _reference_patients() -> list[str]:
    return get_service().reference_patients()


@st.cache_data(show_spinner=False, max_entries=512)
def _row_count(patient_id: str) -> int:
    return get_service().patient_row_count(patient_id)


@st.cache_data(show_spinner="Loading patient history…", max_entries=128)
def _overview(patient_id: str, hour: int, model_name: str) -> dict[str, Any]:
    service = get_service()
    return service.overview(service.load_twin(patient_id, hour, model_name))


@st.cache_data(show_spinner="Scoring the selected ICU hour…", max_entries=128)
def _risk(patient_id: str, hour: int, model_name: str) -> dict[str, Any]:
    service = get_service()
    return service.risk_assessment(service.load_twin(patient_id, hour, model_name))


@st.cache_data(show_spinner=False, max_entries=128)
def _measurements(patient_id: str, hour: int, model_name: str) -> list[dict[str, Any]]:
    service = get_service()
    return service.available_measurements(service.load_twin(patient_id, hour, model_name))


@st.cache_data(show_spinner=False, max_entries=128)
def _vitals_long(
    patient_id: str, hour: int, model_name: str, columns: tuple[str, ...]
) -> pd.DataFrame:
    service = get_service()
    twin = service.load_twin(patient_id, hour, model_name)
    return service.vitals_long(twin, columns)


@st.cache_data(show_spinner="Scoring the historical trajectory…", max_entries=64)
def _trajectory(
    patient_id: str, hour: int, model_name: str, max_hours: int
) -> dict[str, Any]:
    service = get_service()
    twin = service.load_twin(patient_id, hour, model_name)
    return service.risk_trajectory(twin, max_hours=max_hours)


@st.cache_data(show_spinner=False, max_entries=4)
def _scenario_catalogue() -> dict[str, Any]:
    return get_service().scenario_catalogue()


@st.cache_data(show_spinner="Projecting the scenario…", max_entries=64)
def _scenario(
    patient_id: str, hour: int, model_name: str, intervention: str
) -> dict[str, Any]:
    service = get_service()
    twin = service.load_twin(patient_id, hour, model_name)
    return service.scenario(twin, intervention)


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _fmt_value(value: float | None, unit: str | None = None) -> str:
    if value is None:
        return "not recorded"
    if isinstance(value, float) and math.isnan(value):
        return "not recorded"
    if abs(value) >= 1000:
        text = f"{value:,.0f}"
    elif float(value).is_integer():
        text = f"{int(value)}"
    else:
        text = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{text} {unit}" if unit else text


def _fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _chips(items: Sequence[tuple[str, str]]) -> str:
    return "".join(
        f'<span class="ts-chip{" is-" + tone if tone else ""}">{_escape(text)}</span>'
        for text, tone in items
    )


def _kpi(label: str, value: str, meta: str = "", tone: str = "neutral", small: bool = False) -> str:
    classes = ["ts-kpi-value", f"is-{tone}"]
    if small:
        classes.append("is-small")
    return (
        '<div class="ts-kpi">'
        f'<div class="ts-kpi-label">{_escape(label)}</div>'
        f'<div class="{" ".join(classes)}">{_escape(value)}</div>'
        f'<div class="ts-kpi-meta">{_escape(meta)}</div>'
        "</div>"
    )


def _names_list(items: Sequence[str]) -> str:
    return ", ".join(f"<code>{_escape(item)}</code>" for item in items)


def _unavailable(title: str, message: str) -> None:
    st.markdown(
        '<div class="ts-availability">'
        f'<div class="ts-availability-title">{_escape(title)}</div>'
        f'<div class="ts-muted">{_escape(message)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _banner(message: str, tone: str = "") -> None:
    st.markdown(
        f'<div class="ts-banner{" is-" + tone if tone else ""}">{_escape(message)}</div>',
        unsafe_allow_html=True,
    )


def _section_title(title: str, subtitle: str | None = None) -> None:
    subtitle_html = (
        f'<div class="ts-subtitle">{_escape(subtitle)}</div>' if subtitle else ""
    )
    st.markdown(
        f'<div class="ts-eyebrow">{_escape(title)}</div>{subtitle_html}',
        unsafe_allow_html=True,
    )


def _kv_rows(items: Sequence[dict[str, Any]]) -> str:
    rows: list[str] = []
    for item in items:
        value = _fmt_value(item["latest_value"], item["unit"])
        value_class = "ts-kv-value" if item["latest_available"] else "ts-kv-value ts-dim"
        rows.append(
            '<div class="ts-kv">'
            f'<span class="ts-kv-label">{_escape(item["label"])}</span>'
            f'<span class="{value_class}">{_escape(value)}</span>'
            "</div>"
        )
    return "".join(rows)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _sidebar(service: DashboardService) -> dict[str, Any] | None:
    with st.sidebar:
        st.markdown(
            '<div class="ts-brand">Twin<span class="ts-brand-accent">Sepsis</span></div>'
            '<div class="ts-brand-sub">Clinical research dashboard</div>',
            unsafe_allow_html=True,
        )

        status = llm_status()
        references = _reference_patients()

        st.markdown(
            '<div class="ts-eyebrow" style="margin-top:18px">Patient</div>',
            unsafe_allow_html=True,
        )

        options = ["Custom identifier…"] + references
        if "patient_id" not in st.session_state:
            st.session_state["patient_id"] = references[0] if references else "p000001"
        if "quick_patient" not in st.session_state:
            st.session_state["quick_patient"] = options[0]

        def _apply_quick_pick() -> None:
            choice = st.session_state.get("quick_patient")
            if choice in references:
                st.session_state["patient_id"] = choice

        st.selectbox(
            "Reference patient",
            options,
            key="quick_patient",
            on_change=_apply_quick_pick,
            help=(
                "A few identifiers from the local PhysioNet dataset. Any valid "
                "pNNNNNN identifier can be typed below."
            ),
        )
        patient_id = st.text_input(
            "Patient ID",
            key="patient_id",
            help="PhysioNet identifier such as p000001.",
        )

        patient_id = (patient_id or "").strip()
        if not PATIENT_ID_PATTERN.fullmatch(patient_id):
            st.markdown(
                '<div class="ts-dim">Patient IDs look like <code>p000001</code>.</div>',
                unsafe_allow_html=True,
            )
            return None

        try:
            rows = _row_count(patient_id)
        except DashboardError as exc:
            st.error(str(exc))
            return None

        hour = st.slider(
            "ICU hour",
            min_value=1,
            max_value=rows,
            value=rows,
            key=f"hour::{patient_id}",
            help=f"This patient has {rows} recorded hourly rows.",
        )

        st.markdown(
            '<div class="ts-eyebrow" style="margin-top:14px">Risk model</div>',
            unsafe_allow_html=True,
        )
        model_name = st.radio(
            "Risk model",
            options=list(MODEL_CHOICES),
            format_func=lambda key: MODEL_LABELS[key],
            key="risk_model",
            label_visibility="collapsed",
        )
        available, reason = service.model_availability(model_name)
        if not available:
            st.warning(reason)
        st.markdown(
            f'<div class="ts-dim">{_escape(MODEL_SUMMARIES[model_name])}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="ts-eyebrow" style="margin-top:18px">Assistant</div>',
            unsafe_allow_html=True,
        )
        if status["available"]:
            use_llm = st.toggle(
                "Use LLM planner",
                value=True,
                help=(
                    f"{status['provider']} · {status['model']}. Turn this off to "
                    "use the deterministic tools-only workflow."
                ),
            )
        else:
            use_llm = False
            st.markdown(
                '<div class="ts-dim">No LLM configured — deterministic '
                "tools-only mode.</div>",
                unsafe_allow_html=True,
            )

        with st.expander("Runtime details"):
            try:
                root = str(service.dataset_root)
            except DashboardError:
                root = "not configured"
            loaded = ", ".join(service.models.loaded) or "none yet"
            st.markdown(
                '<div class="ts-mono">'
                f"Python {platform.python_version()}<br>"
                f"Dataset root: {_escape(root)}<br>"
                f"Models cached: {_escape(loaded)}<br>"
                f"LLM provider: {_escape(status['provider'] or 'none')}"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="ts-dim" style="margin-top:8px">The LLM credential is '
                "read from the environment and never displayed.</div>",
                unsafe_allow_html=True,
            )

    return {
        "patient_id": patient_id,
        "hour": int(hour),
        "rows": int(rows),
        "model_name": model_name,
        "use_llm": bool(use_llm),
        "llm_status": status,
    }


# ---------------------------------------------------------------------------
# Header and headline metrics
# ---------------------------------------------------------------------------


def _render_header(selection: dict[str, Any], capabilities: dict[str, Any]) -> None:
    left, right = st.columns([3.1, 1.2], vertical_alignment="center")
    with left:
        st.markdown(
            '<div class="ts-eyebrow">Sepsis risk review · research prototype</div>'
            f'<div class="ts-title">{_escape(selection["patient_id"])}</div>'
            f'<div class="ts-subtitle">ICU hour {selection["hour"]} of '
            f'{selection["rows"]} recorded hours · '
            f'{_escape(capabilities["dataset"])}</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="ts-banner" style="text-align:left">Research-model '
            "estimates only. This is not a clinical diagnosis, a treatment "
            "recommendation, or a validated decision-support tool.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        _chips(
            [
                (capabilities["model_label"], "teal"),
                (capabilities["model"], "blue"),
                (f"threshold {capabilities['threshold']:.3f}", "amber"),
                (f"{selection['hour']} h history", ""),
            ]
        ),
        unsafe_allow_html=True,
    )


def _render_headline(risk: dict[str, Any], hour: int) -> None:
    trend = risk.get("trend") or {}
    direction = trend.get("direction")
    window = trend.get("window_hours") or 0
    delta = trend.get("delta")

    if direction == "insufficient_history" or delta is None:
        trend_value, trend_meta, trend_tone = "Insufficient history", "one scored hour so far", "neutral"
    else:
        arrow = {"rising": "▲", "falling": "▼", "stable": "■"}.get(direction, "")
        trend_value = f"{arrow} {delta * 100:+.1f} pts"
        trend_meta = f"over {window} h · hour t−{window} to t"
        trend_tone = {"rising": "alert", "falling": "ok"}.get(direction, "neutral")

    columns = st.columns(4)
    with columns[0]:
        st.markdown(
            _kpi(
                "Current risk",
                _fmt_pct(risk["risk"]),
                f"model estimate at ICU hour {hour}",
            ),
            unsafe_allow_html=True,
        )
    with columns[1]:
        st.markdown(
            _kpi(
                "Frozen threshold",
                f"{risk['threshold']:.3f}",
                "validation-selected, never retuned here",
            ),
            unsafe_allow_html=True,
        )
    with columns[2]:
        exceeds = risk["exceeds_threshold"]
        st.markdown(
            _kpi(
                "Threshold decision",
                "At or above" if exceeds else "Below",
                f"risk {'≥' if exceeds else '<'} {risk['threshold']:.3f}",
                tone="alert" if exceeds else "ok",
                small=True,
            ),
            unsafe_allow_html=True,
        )
    with columns[3]:
        st.markdown(
            _kpi("Risk trend", trend_value, trend_meta, tone=trend_tone, small=True),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Section A — patient overview
# ---------------------------------------------------------------------------


def _render_overview(overview: dict[str, Any]) -> None:
    left, right = st.columns([1, 1.25])

    with left:
        with st.container(border=True):
            _section_title("Patient overview", "Only fields present in the local patient file are shown.")
            st.markdown(
                _kv_rows(overview["demographics"]),
                unsafe_allow_html=True,
            )
            recorded = [item for item in overview["measurements"] if item["latest_available"]]
            st.markdown(
                f'<div class="ts-eyebrow" style="margin-top:16px">Latest recorded '
                f"measurements · {len(recorded)}</div>",
                unsafe_allow_html=True,
            )
            if recorded:
                st.markdown(_kv_rows(recorded), unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="ts-muted">No measurements are recorded at this '
                    "hour.</div>",
                    unsafe_allow_html=True,
                )

    with right:
        with st.container(border=True):
            missing = overview["missing"]
            _section_title(
                "Data completeness",
                f"{overview['hours']} hourly rows · {missing['feature_columns']} PhysioNet features",
            )
            columns = st.columns(3)
            with columns[0]:
                st.markdown(
                    _kpi(
                        "Cells missing",
                        _fmt_pct(missing["share_missing"], 0),
                        f"{missing['cells_missing']:,} of {missing['cells_total']:,}",
                        small=True,
                    ),
                    unsafe_allow_html=True,
                )
            with columns[1]:
                st.markdown(
                    _kpi(
                        "Features observed",
                        f"{missing['observed_columns']}/{missing['feature_columns']}",
                        "at least one value in history",
                        small=True,
                    ),
                    unsafe_allow_html=True,
                )
            with columns[2]:
                st.markdown(
                    _kpi(
                        "Never recorded",
                        str(len(missing["never_recorded"])),
                        "columns with no values",
                        tone="neutral",
                        small=True,
                    ),
                    unsafe_allow_html=True,
                )

            frame = pd.DataFrame(
                [
                    {
                        "Measurement": item["label"],
                        "Latest value": _fmt_value(item["latest_value"], item["unit"]),
                        "Recorded": (
                            "at this hour"
                            if item["latest_available"]
                            else "earlier hours only"
                            if item["available"]
                            else "never"
                        ),
                        "Coverage": (
                            f"{item['observed']}/{item['total']} h · "
                            f"{_fmt_pct(item['missing_share'], 0)} missing"
                        ),
                    }
                    for item in overview["measurements"]
                ]
            )
            if frame.empty:
                st.info("This patient file contains none of the dashboard measurements.")
            else:
                st.dataframe(frame, hide_index=True, height=260)

            if missing["never_recorded"]:
                with st.expander(
                    f"Features never recorded for this patient ({len(missing['never_recorded'])})"
                ):
                    st.markdown(
                        f'<div class="ts-muted">{_names_list(missing["never_recorded"])}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "These columns exist in the PhysioNet schema but have no "
                        "values in this patient's history. They are reported, not imputed."
                    )


# ---------------------------------------------------------------------------
# Section B/C — risk assessment and historical trajectory
# ---------------------------------------------------------------------------


def _render_risk(risk: dict[str, Any]) -> None:
    with st.container(border=True):
        left, right = st.columns([1.15, 1])

        with left:
            _section_title(
                "Sepsis risk assessment",
                "Research-model estimate for the scored ICU hour.",
            )
            width = max(0.0, min(1.0, risk["risk"]))
            threshold = max(0.0, min(1.0, risk["threshold"]))
            st.markdown(
                '<div class="ts-meter">'
                f'<div class="ts-meter-fill" style="width:{width * 100:.2f}%"></div>'
                f'<div class="ts-meter-mark" style="left:calc({threshold * 100:.2f}% - 1px)"></div>'
                "</div>"
                '<div class="ts-meter-scale">'
                "<span>0%</span>"
                f'<span>threshold {risk["threshold"]:.3f}</span>'
                "<span>100%</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            state = "at or above" if risk["exceeds_threshold"] else "below"
            st.markdown(
                f'<div class="ts-muted" style="margin-top:12px">Model-estimated risk '
                f'<b>{_fmt_pct(risk["risk"], 2)}</b> is {state} the frozen threshold of '
                f'<b>{risk["threshold"]:.3f}</b>. This is a research-model estimate, '
                "not a diagnosis.</div>",
                unsafe_allow_html=True,
            )

            if risk.get("components"):
                components = risk["components"]
                st.markdown(
                    '<div class="ts-eyebrow" style="margin-top:16px">Ensemble '
                    "components</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    _chips(
                        [
                            (f"XGBoost {components['xgboost']:.3f} × {components['xgboost_weight']:.2f}", "blue"),
                            (f"GRU {components['gru']:.3f} × {components['gru_weight']:.2f}", "teal"),
                            (f"blend {risk['risk']:.3f}", "amber"),
                        ]
                    ),
                    unsafe_allow_html=True,
                )
                st.altair_chart(
                    charts.component_chart(components),
                    width="stretch",
                )

        with right:
            _section_title("Model and decision context")
            st.markdown(
                '<div class="ts-muted">'
                f"<b>Model</b> · {_escape(risk['risk_model'])}<br>"
                f"<b>Dataset</b> · {_escape(risk['dataset'])}<br>"
                f"<b>Scored ICU hour</b> · {risk['history_index'] + 1} (0-based history index {risk['history_index']})<br>"
                f"<b>Threshold</b> · {risk['threshold']:.3f} (frozen)<br>"
                f"<b>Horizon</b> · {_escape(risk.get('prediction_horizon') or 'not reported')}"
                "</div>",
                unsafe_allow_html=True,
            )
            if risk.get("artifact"):
                st.markdown(
                    f'<div class="ts-mono" style="margin-top:10px">{_escape(risk["artifact"])}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<div class="ts-dim" style="margin-top:10px">Both backends score the '
                "current ICU hour only. No multi-hour forecast is produced, and no "
                "causal treatment effect is estimated.</div>",
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        _section_title(
            "Model drivers for this hour",
            "Feature contributions are model attributions, not causal effects.",
        )
        if risk["explanation_available"] and risk.get("drivers"):
            frame = pd.DataFrame(
                [
                    {
                        "Feature": driver["feature"],
                        "Label": driver["label"],
                        "Value": pd.NA if driver["value"] is None else driver["value"],
                        "Contribution": driver["contribution"],
                        "Direction": driver["direction"],
                    }
                    for driver in risk["drivers"]
                ]
            )
            st.dataframe(
                frame,
                hide_index=True,
                column_config={
                    "Contribution": st.column_config.NumberColumn(format="%.4f"),
                    "Value": st.column_config.NumberColumn(format="%.3f"),
                },
            )
            units = risk.get("explanation_units")
            if units:
                st.caption(f"Contribution units: {units}.")
        else:
            _unavailable(
                "Feature attribution unavailable",
                risk.get("explanation_note")
                or "The selected model does not expose feature contributions.",
            )


def _render_trajectory(
    selection: dict[str, Any], service: DashboardService
) -> None:
    total = selection["rows"]
    options = trajectory_window_options(total)
    default = max([value for value in options if value <= 24] or [options[0]])

    with st.container(border=True):
        _section_title(
            "Historical risk trajectory",
            "Each hour is re-scored from that hour's own data prefix.",
        )

        window = st.select_slider(
            "History window",
            options=options,
            value=default,
            format_func=lambda value: (
                f"all {total} h" if value >= total else f"{value} h"
            ),
            key=f"window::{selection['patient_id']}::{selection['hour']}",
        )

        trajectory = _trajectory(
            selection["patient_id"],
            selection["hour"],
            selection["model_name"],
            int(window),
        )
        frame = service.trajectory_frame(trajectory)

        st.altair_chart(
            charts.risk_trajectory_chart(
                frame, trajectory["threshold"], selection["hour"]
            ),
            width="stretch",
        )

        st.markdown(
            '<div class="ts-muted">Historical prefix-based scores from hour '
            f'{trajectory["from_hour"]} to {trajectory["to_hour"]}. Every point uses '
            "only the data available at that hour; earlier points do not change when "
            "later hours exist. These are <b>not forecasts</b> and nothing is "
            "extrapolated beyond the scored hour.</div>",
            unsafe_allow_html=True,
        )
        if window >= 168:
            st.caption(
                "Long windows re-score many hourly prefixes; the first render of a "
                "window this large can take a while."
            )

        with st.expander("Trajectory values"):
            table = frame.copy()
            table["risk"] = table["risk"].map(lambda value: round(float(value), 4))
            table.columns = ["ICU hour", "Risk"]
            st.dataframe(table, hide_index=True, height=240)


# ---------------------------------------------------------------------------
# Section D — vitals
# ---------------------------------------------------------------------------


def _render_vitals(selection: dict[str, Any]) -> None:
    available = _measurements(
        selection["patient_id"], selection["hour"], selection["model_name"]
    )

    with st.container(border=True):
        _section_title(
            "Physiological measurements",
            "Only measurements present in this patient's data are listed.",
        )

        if not available:
            st.info(
                "This patient has none of the dashboard's measurements recorded in "
                "the loaded history."
            )
            return

        labels = {
            item["column"]: (
                f"{item['label']} ({item['unit']})" if item["unit"] else item["label"]
            )
            for item in available
        }
        preferred = ("HR", "MAP", "Temp", "Resp", "O2Sat")
        default = [column for column in preferred if column in labels][:4]

        chosen = st.multiselect(
            "Measurements to plot",
            options=list(labels),
            default=default,
            format_func=lambda column: labels[column],
            key=(
                f"vitals::{selection['patient_id']}::{selection['hour']}"
            ),
        )

        if not chosen:
            st.info("Select at least one measurement to plot its history.")
            return

        long = _vitals_long(
            selection["patient_id"],
            selection["hour"],
            selection["model_name"],
            tuple(chosen),
        )
        st.altair_chart(
            charts.vitals_chart(long, selection["hour"], columns=2),
            width="stretch",
        )
        st.markdown(
            '<div class="ts-muted">Gaps are hours where the measurement was not '
            "recorded. Values are never imputed, interpolated or carried backwards. "
            "The dashed line marks the scored ICU hour.</div>",
            unsafe_allow_html=True,
        )

        availability = pd.DataFrame(
            [
                {
                    "Measurement": labels[item["column"]],
                    "Status": "recorded" if item["observed"] else "not recorded",
                    "Observed hours": f"{item['observed']}/{item['total']}",
                    "Missing": _fmt_pct(item["missing_share"], 0),
                }
                for item in available
                if item["column"] in chosen
            ]
        )
        with st.expander("Measurement coverage"):
            st.dataframe(availability, hide_index=True)


# ---------------------------------------------------------------------------
# Section E — agentic assistant
# ---------------------------------------------------------------------------


def _render_agent_response(response: dict[str, Any]) -> None:
    used = response.get("tools_used") or []
    failed = [item for item in used if not item.get("ok")]

    summary_note = (
        " · deterministic system-generated summary (not an LLM response)"
        if response.get("summary_source") == "deterministic_fallback"
        else ""
    )
    st.caption(
        f"mode: {response.get('mode')}{summary_note} · {len(used)} tool call(s) · "
        f"{len(failed)} unavailable or failed · model "
        f"{(response.get('prediction_source') or {}).get('model', 'n/a')}"
    )

    with st.expander(f"Tool execution ({len(used)})"):
        rows = [
            {
                "Tool": item.get("tool"),
                "Status": "ok" if item.get("ok") else "unavailable/failed",
                "Origin": item.get("origin"),
                "Error": item.get("error_type") or "",
                "Arguments": json.dumps(item.get("arguments") or {}, sort_keys=True),
            }
            for item in used
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        else:
            st.caption("No tools were executed in this run.")

    if response.get("top_drivers"):
        with st.expander("Model drivers returned by the tools"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Feature": driver.get("feature"),
                            "Label": driver.get("label"),
                            "Value": driver.get("value"),
                            "Contribution": driver.get("contribution"),
                        }
                        for driver in response["top_drivers"]
                    ]
                ),
                hide_index=True,
            )

    limitations = response.get("limitations") or []
    caveats = response.get("caveats") or []
    if limitations or caveats:
        with st.expander("Limitations and caveats"):
            if limitations:
                st.markdown(
                    f'<div class="ts-dim">Limitations</div>'
                    f'<div class="ts-muted">{_names_list(limitations)}</div>',
                    unsafe_allow_html=True,
                )
            for caveat in caveats:
                st.markdown(f"- {caveat}")


def _answer(
    selection: dict[str, Any],
    prompt: str,
    *,
    use_llm: bool,
) -> None:
    service = get_service()
    patient_id = selection["patient_id"]
    hour = selection["hour"]
    model_name = selection["model_name"]

    history: list[dict[str, Any]] = st.session_state["chat_messages"]
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        twin = service.load_twin(patient_id, hour, model_name)
    except DashboardError as exc:
        with st.chat_message("assistant"):
            st.error(str(exc))
        return

    with st.chat_message("assistant"):
        client = None
        response: dict[str, Any] | None = None
        failure: str | None = None

        try:
            with st.spinner("Running the agent's tools…"):
                if use_llm:
                    client = create_client(LLMConfig.from_env())
                response = service.assistant(twin, prompt, llm=client)
        except (LLMError, AgentError) as exc:
            failure = str(exc)
        finally:
            if client is not None:
                client.close()

        if failure is not None:
            st.error(f"The LLM planner could not complete this request: {failure}")
            st.caption(
                "No natural-language answer was generated. The deterministic tool "
                "sequence is shown below instead."
            )
            try:
                response = service.assistant(twin, prompt, llm=None)
            except (AgentError, DashboardError) as exc:
                st.error(f"The deterministic tools also failed: {exc}")
                history.append(
                    {
                        "role": "assistant",
                        "content": f"The LLM planner failed: {failure}",
                    }
                )
                return

        assert response is not None
        text = response.get("risk_summary") or "No risk summary was produced."
        if failure is not None or response.get("mode") == "tools_only":
            text = (
                "**Deterministic tool output** (no LLM planner was used).\n\n" + text
            )
        st.markdown(text)
        _render_agent_response(response)

        history.append(
            {
                "role": "assistant",
                "content": text,
                "response": response,
            }
        )


def _render_assistant(selection: dict[str, Any], capabilities: dict[str, Any]) -> None:
    with st.container(border=True):
        _section_title(
            "Agentic assistant",
            "Answers are assembled from tool outputs; the assistant cannot invent numbers.",
        )

        if not capabilities["explanation"]:
            _unavailable(
                "Feature attribution is unavailable for this model",
                capabilities["explanation_note"] or "",
            )
            st.write("")
        if not capabilities["simulation"]:
            _unavailable(
                "Intervention simulation is unavailable for this model",
                capabilities["simulation_note"] or "",
            )
            st.write("")

        state_key = (
            f"{selection['patient_id']}|{selection['hour']}|{selection['model_name']}"
        )
        if st.session_state.get("chat_key") != state_key:
            st.session_state["chat_key"] = state_key
            st.session_state["chat_messages"] = []

        status = selection["llm_status"]
        can_use_llm = bool(status["available"] and selection["use_llm"])

        if can_use_llm:
            st.markdown(
                _chips(
                    [
                        (f"{status['provider']} · {status['model']}", "teal"),
                        ("LLM plans tools", ""),
                    ]
                ),
                unsafe_allow_html=True,
            )
        else:
            reason = status["reason"] or "the LLM planner is switched off in the sidebar"
            st.markdown(
                '<div class="ts-banner">Deterministic tools-only mode. '
                f"<span class=\"ts-dim\">{_escape(reason)}</span></div>",
                unsafe_allow_html=True,
            )

        for message in st.session_state.get("chat_messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("response"):
                    _render_agent_response(message["response"])

        if can_use_llm:
            example_columns = st.columns(4)
            for index, (label, example) in enumerate(EXAMPLE_PROMPTS):
                with example_columns[index]:
                    if st.button(
                        label,
                        key=f"example::{index}",
                        width="stretch",
                        help=example,
                    ):
                        st.session_state["queued_prompt"] = example

            queued = st.session_state.pop("queued_prompt", None)
            prompt = st.chat_input("Ask about this patient…") or queued
            if prompt:
                _answer(selection, prompt, use_llm=True)
        else:
            st.caption(
                "Without an LLM planner, the assistant runs the fixed deterministic "
                "tool sequence: patient state, risk, attribution (when supported), "
                "trajectory, clinical scores and scenarios."
            )
            if st.button("Run deterministic tool sequence", type="primary"):
                _answer(
                    selection,
                    "Deterministic tool sequence",
                    use_llm=False,
                )


# ---------------------------------------------------------------------------
# Section F — scenarios
# ---------------------------------------------------------------------------


def _render_scenarios(
    selection: dict[str, Any],
    capabilities: dict[str, Any],
    risk: dict[str, Any],
) -> None:
    with st.container(border=True):
        _section_title(
            "Digital twin · intervention scenarios",
            "Model-based scenarios, not causal treatment effects.",
        )

        if not capabilities["simulation"]:
            _unavailable(
                "Intervention simulation is unavailable for this model",
                capabilities["simulation_note"] or "",
            )
            st.markdown(
                '<div class="ts-muted" style="margin-top:12px">No scenario is run, '
                "no projection is produced, and no treatment effect is implied. "
                "Enabling this would require validating counterfactual re-scoring of "
                "the blend, which this repository does not do.</div>",
                unsafe_allow_html=True,
            )
            return

        catalogue = _scenario_catalogue()
        interventions = catalogue.get("interventions") or []
        labels = {item["name"]: item["label"] for item in interventions}

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Intervention": item["label"],
                        "Key": item["name"],
                        "Effect assumptions": item["effect_status"],
                        "Projection available": item["available"],
                    }
                    for item in interventions
                ]
            ),
            hide_index=True,
        )

        choice = st.selectbox(
            "Scenario",
            options=list(labels),
            format_func=lambda name: labels[name],
            key=f"scenario::{selection['patient_id']}::{selection['hour']}",
        )

        result = _scenario(
            selection["patient_id"],
            selection["hour"],
            selection["model_name"],
            choice,
        )

        if result.get("available") and result.get("projected_risk"):
            projected = result["projected_risk"]
            table = pd.DataFrame(
                [
                    {
                        "Horizon": f"+{horizon} h",
                        "Projected risk": projected.get(str(horizon)),
                        "Envelope low": (result["bounds"].get(str(horizon)) or {}).get("low"),
                        "Envelope high": (result["bounds"].get(str(horizon)) or {}).get("high"),
                    }
                    for horizon in result.get("horizons_hours", [1, 3, 6])
                ]
            )
            st.altair_chart(
                charts.scenario_chart(
                    result,
                    threshold=float(risk["threshold"]),
                    current_risk=float(risk["risk"]),
                ),
                width="stretch",
            )
            st.dataframe(
                table,
                hide_index=True,
                column_config={
                    "Projected risk": st.column_config.NumberColumn(format="%.3f"),
                    "Envelope low": st.column_config.NumberColumn(format="%.3f"),
                    "Envelope high": st.column_config.NumberColumn(format="%.3f"),
                },
            )
            st.markdown(
                '<div class="ts-muted">The envelope is the configured assumption '
                "range, not a confidence interval. Nothing here says that the "
                "intervention improves outcomes.</div>",
                unsafe_allow_html=True,
            )

            assumptions = result.get("assumptions") or []
            if assumptions:
                with st.expander(f"Effect assumptions ({len(assumptions)})"):
                    st.dataframe(pd.DataFrame(assumptions), hide_index=True)
        else:
            st.warning(result.get("reason") or "No projection was produced.")
            baseline = result.get("baseline_no_action") or {}
            if baseline:
                st.markdown(
                    '<div class="ts-muted">No-action reference (observed values '
                    "carried forward, re-scored by the same model):</div>",
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Horizon": f"+{key} h", "No-action risk": value}
                            for key, value in baseline.items()
                        ]
                    ),
                    hide_index=True,
                )
            unresolved = result.get("unresolved") or []
            if unresolved:
                st.markdown(
                    f'<div class="ts-dim">Unresolved assumptions: '
                    f"{_names_list(unresolved)}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown(
            f'<div class="ts-banner">{_escape(result.get("disclaimer") or "")}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def _render_footer() -> None:
    st.markdown(
        '<div class="ts-footer">'
        "TwinSepsis research prototype. Risk models are trained on PhysioNet "
        "Challenge 2019; no MIMIC-IV model exists in this repository. Risk is "
        "scored for the selected ICU hour only — there is no multi-hour forecast. "
        "Feature attributions are model attributions, not causal effects, and "
        "scenarios are model-based projections, not treatment effects or clinical "
        "advice. A clinician must review every case."
        "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="TwinSepsis · clinical research dashboard",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(theme.CSS, unsafe_allow_html=True)

    service = get_service()
    selection = _sidebar(service)
    if selection is None:
        st.markdown(
            '<div class="ts-title" style="margin-top:24px">Select a patient</div>'
            '<div class="ts-muted">Choose one of the reference patients or type a '
            "valid PhysioNet identifier in the sidebar.</div>",
            unsafe_allow_html=True,
        )
        return

    try:
        model = service.model(selection["model_name"])
        capabilities = service.capabilities(model)
        overview = _overview(
            selection["patient_id"], selection["hour"], selection["model_name"]
        )
        risk = _risk(
            selection["patient_id"], selection["hour"], selection["model_name"]
        )
    except DashboardError as exc:
        st.error(str(exc))
        _banner(
            "The selected patient or model could not be loaded. Nothing was "
            "substituted or fabricated; pick another patient or model in the sidebar."
        )
        return

    _render_header(selection, capabilities)
    _render_headline(risk, selection["hour"])

    overview_tab, risk_tab, vitals_tab, assistant_tab, scenario_tab = st.tabs(
        [
            "Patient overview",
            "Risk assessment",
            "Physiological data",
            "AI assistant",
            "Scenarios",
        ]
    )

    with overview_tab:
        _render_overview(overview)
    with risk_tab:
        _render_risk(risk)
        _render_trajectory(selection, service)
    with vitals_tab:
        _render_vitals(selection)
    with assistant_tab:
        _render_assistant(selection, capabilities)
    with scenario_tab:
        _render_scenarios(selection, capabilities, risk)

    _render_footer()


__all__ = ["main"]
