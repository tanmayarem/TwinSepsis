"""Altair chart builders for the TwinSepsis dashboard.

Presentation only: every number plotted here comes from
:mod:`twinsepsis.dashboard.service`, which in turn delegates to the agent
backend.  Nothing in this module scores a patient.
"""

from __future__ import annotations

from typing import Any, Sequence

import altair as alt
import pandas as pd

from . import theme


def _styled(
    chart: alt.TopLevelMixin, height: int | None = None
) -> alt.TopLevelMixin:
    if height is not None:
        chart = chart.properties(height=height)
    return (
        chart
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=theme.MUTED,
            titleColor=theme.DIM,
            gridColor=theme.GRID,
            domainColor=theme.BORDER,
            tickColor=theme.BORDER,
            labelFontSize=11,
            titleFontSize=11,
        )
        .configure_legend(
            labelColor=theme.MUTED,
            titleColor=theme.DIM,
            orient="bottom",
            direction="horizontal",
            labelFontSize=11,
        )
    )


def risk_trajectory_chart(
    frame: pd.DataFrame,
    threshold: float,
    current_hour: int,
    *,
    height: int = 290,
) -> alt.TopLevelMixin:
    """Historical prefix-based risk with threshold and current-hour markers."""

    if frame.empty:
        raise ValueError("risk_trajectory_chart needs at least one scored hour")

    base = alt.Chart(frame)
    x = alt.X(
        "icu_hour:Q",
        title="ICU hour",
        axis=alt.Axis(format="d", tickMinStep=1),
    )
    y = alt.Y(
        "risk:Q",
        title="Model-estimated risk",
        scale=alt.Scale(domain=[0.0, 1.0], clamp=True),
        axis=alt.Axis(format="%"),
    )
    tooltip = [
        alt.Tooltip("icu_hour:Q", title="ICU hour", format="d"),
        alt.Tooltip("risk:Q", title="Risk", format=".3f"),
    ]

    area = base.mark_area(color=theme.TEAL, opacity=0.14)
    line = base.mark_line(
        color=theme.TEAL,
        strokeWidth=2.2,
        point=alt.OverlayMarkDef(size=34, filled=True, color=theme.TEAL, opacity=0.9),
    )
    series = (area + line).encode(x=x, y=y, tooltip=tooltip)

    threshold_rule = alt.Chart(frame).mark_rule(
        color=theme.AMBER, strokeDash=[5, 4], strokeWidth=1.4
    ).encode(y=alt.datum(threshold))
    threshold_label = alt.Chart(frame).mark_text(
        align="left",
        baseline="top",
        dx=4,
        dy=2,
        color=theme.AMBER,
        fontSize=10,
    ).encode(
        x=alt.datum(float(frame["icu_hour"].min())),
        y=alt.datum(threshold),
        text=alt.value(f"threshold {threshold:.2f}"),
    )

    current_rule = alt.Chart(frame).mark_rule(
        color=theme.BLUE, strokeDash=[3, 3], strokeWidth=1
    ).encode(x=alt.datum(float(current_hour)))
    current_label = alt.Chart(frame).mark_text(
        align="right",
        baseline="bottom",
        dx=-6,
        dy=-4,
        color=theme.BLUE,
        fontSize=10,
    ).encode(
        x=alt.datum(float(current_hour)),
        y=alt.datum(1.0),
        text=alt.value(f"current hour {current_hour}"),
    )

    latest = alt.Chart(frame.tail(1)).mark_circle(
        size=130,
        color=theme.BLUE,
        stroke=theme.NAVY,
        strokeWidth=2,
    ).encode(x=x, y=y, tooltip=tooltip)

    chart = alt.layer(
        series,
        threshold_rule,
        threshold_label,
        current_rule,
        current_label,
        latest,
    )
    return _styled(chart, height)


def vitals_chart(
    long_frame: pd.DataFrame,
    current_hour: int,
    *,
    columns: int = 2,
    height: int = 128,
) -> alt.TopLevelMixin:
    """Small-multiples chart of measured vitals.

    Missing values stay in the frame as nulls, so Vega-Lite breaks the line
    instead of interpolating across a gap.
    """

    if long_frame.empty:
        raise ValueError("vitals_chart needs at least one measurement")

    base = alt.Chart(long_frame).mark_line(
        color=theme.TEAL,
        strokeWidth=1.6,
        point=alt.OverlayMarkDef(size=20, filled=True, color=theme.TEAL, opacity=0.75),
    ).encode(
        x=alt.X("icu_hour:Q", title="ICU hour"),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False, nice=True)),
        tooltip=[
            alt.Tooltip("icu_hour:Q", title="ICU hour", format="d"),
            alt.Tooltip("value:Q", title="Value", format=".2f"),
            alt.Tooltip("measurement:N", title="Measurement"),
        ],
    )

    current_rule = alt.Chart(long_frame).mark_rule(
        color=theme.BLUE, strokeDash=[3, 3], strokeWidth=1
    ).encode(x=alt.datum(float(current_hour)))

    chart = alt.layer(base, current_rule).properties(width=300, height=height)
    chart = chart.facet(
        facet=alt.Facet("measurement:N", title=None), columns=columns
    ).resolve_scale(y="independent")
    return _styled(chart)


def component_chart(components: dict[str, Any], *, height: int = 112) -> alt.TopLevelMixin:
    """Weighted contribution of each ensemble component to the blend."""

    frame = pd.DataFrame(
        {
            "component": ["XGBoost", "GRU"],
            "weight": [components["xgboost_weight"], components["gru_weight"]],
            "probability": [components["xgboost"], components["gru"]],
        }
    )
    frame["contribution"] = frame["weight"] * frame["probability"]

    bars = alt.Chart(frame).mark_bar(cornerRadius=4, height=18).encode(
        y=alt.Y("component:N", title=None, sort=["XGBoost", "GRU"]),
        x=alt.X(
            "contribution:Q",
            title="weight × component probability",
            scale=alt.Scale(domain=[0.0, 1.0], clamp=True),
        ),
        color=alt.Color(
            "component:N",
            legend=None,
            scale=alt.Scale(
                domain=["XGBoost", "GRU"], range=[theme.BLUE, theme.TEAL]
            ),
        ),
        tooltip=[
            alt.Tooltip("component:N", title="Component"),
            alt.Tooltip("probability:Q", title="Probability", format=".3f"),
            alt.Tooltip("weight:Q", title="Weight", format=".2f"),
            alt.Tooltip("contribution:Q", title="weight × probability", format=".3f"),
        ],
    )
    labels = alt.Chart(frame).mark_text(
        align="left", dx=6, color=theme.MUTED, fontSize=11
    ).encode(
        y=alt.Y("component:N", sort=["XGBoost", "GRU"]),
        x="contribution:Q",
        text=alt.Text("contribution:Q", format=".3f"),
    )
    return _styled(bars + labels, height)


def scenario_chart(
    result: dict[str, Any],
    *,
    threshold: float,
    current_risk: float,
    height: int = 250,
) -> alt.TopLevelMixin | None:
    """Projected scenario risk per horizon with its assumption envelope."""

    projected = result.get("projected_risk") or {}
    bounds = result.get("bounds") or {}
    horizons = [int(hour) for hour in result.get("horizons_hours") or (1, 3, 6)]

    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        key = str(horizon)
        value = projected.get(key)
        if value is None:
            continue
        envelope = bounds.get(key) or {}
        rows.append(
            {
                "horizon": f"+{horizon} h",
                "risk": float(value),
                "low": envelope.get("low"),
                "high": envelope.get("high"),
            }
        )

    if not rows:
        return None

    frame = pd.DataFrame(rows)
    order = [f"+{horizon} h" for horizon in horizons if f"+{horizon} h" in set(frame["horizon"])]

    bars = alt.Chart(frame).mark_bar(size=52, cornerRadius=5, color=theme.BLUE).encode(
        x=alt.X("horizon:N", title="Hours after the scored hour", sort=order),
        y=alt.Y(
            "risk:Q",
            title="Projected model risk",
            scale=alt.Scale(domain=[0.0, 1.0], clamp=True),
            axis=alt.Axis(format="%"),
        ),
        tooltip=[
            alt.Tooltip("horizon:N", title="Horizon"),
            alt.Tooltip("risk:Q", title="Projected risk", format=".3f"),
            alt.Tooltip("low:Q", title="Envelope low", format=".3f"),
            alt.Tooltip("high:Q", title="Envelope high", format=".3f"),
        ],
    )
    envelope = alt.Chart(frame).mark_rule(color=theme.TEAL, strokeWidth=2).encode(
        x=alt.X("horizon:N", sort=order),
        y=alt.Y("low:Q"),
        y2=alt.Y2("high:Q"),
    )
    threshold_rule = alt.Chart(frame).mark_rule(
        color=theme.AMBER, strokeDash=[5, 4], strokeWidth=1.2
    ).encode(y=alt.datum(threshold))
    baseline_rule = alt.Chart(frame).mark_rule(
        color=theme.MUTED, strokeDash=[2, 3], strokeWidth=1
    ).encode(y=alt.datum(current_risk))

    return _styled(bars + envelope + threshold_rule + baseline_rule, height)


__all__ = [
    "component_chart",
    "risk_trajectory_chart",
    "scenario_chart",
    "vitals_chart",
]
