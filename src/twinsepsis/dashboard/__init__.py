"""Streamlit research dashboard for the TwinSepsis agent.

The package is split so the UI stays thin:

* :mod:`twinsepsis.dashboard.service` is a plain-Python adapter over the
  existing patient source, risk models, tools and :class:`SepsisAgent`.  It has
  no Streamlit import and is unit-tested directly.
* :mod:`twinsepsis.dashboard.app` renders that adapter in Streamlit.

Nothing here retrains a model, changes a threshold, or re-implements inference.
"""

from .service import (
    MODEL_CHOICES,
    MODEL_LABELS,
    DashboardError,
    DashboardService,
    MeasurementSpec,
    ModelCache,
    llm_status,
    normalize_model_name,
)

__all__ = [
    "MODEL_CHOICES",
    "MODEL_LABELS",
    "DashboardError",
    "DashboardService",
    "MeasurementSpec",
    "ModelCache",
    "llm_status",
    "normalize_model_name",
]
