"""Visual theme for the TwinSepsis dashboard.

The palette and the CSS block live here so the page module stays about content.
Colours are deliberately restrained: a dark navy clinical base with teal and
blue accents, amber for thresholds/warnings and red only for threshold breaches.
"""

from __future__ import annotations

NAVY = "#070E1A"
SIDEBAR = "#0A1322"
PANEL = "#0E1A2B"
PANEL_ALT = "#112034"
BORDER = "#1D2E47"
TEXT = "#E9F1FB"
MUTED = "#93A7C0"
DIM = "#6B8099"
TEAL = "#2FD6C0"
BLUE = "#48B4F5"
AMBER = "#F0B13C"
RED = "#F27A7A"
GREEN = "#4ADE9B"
GRID = "#182740"

CSS = f"""
<style>
:root {{
  color-scheme: dark;
  --ts-text: {TEXT};
  --ts-muted: {MUTED};
  --ts-dim: {DIM};
  --ts-border: {BORDER};
  --ts-panel: {PANEL};
  --ts-teal: {TEAL};
  --ts-blue: {BLUE};
  --ts-amber: {AMBER};
  --ts-red: {RED};
  --ts-green: {GREEN};
}}

.stApp {{
  background:
    radial-gradient(1100px 520px at 12% -12%, #12253C 0%, rgba(7, 14, 26, 0) 62%),
    radial-gradient(900px 420px at 96% 4%, #0E2233 0%, rgba(7, 14, 26, 0) 58%),
    {NAVY};
  color: {TEXT};
}}

header[data-testid="stHeader"] {{
  background: transparent;
  height: 0;
}}

#MainMenu, footer {{ visibility: hidden; }}

.block-container {{
  padding-top: 1.6rem;
  padding-bottom: 2.4rem;
  max-width: 1480px;
}}

/* Font family is left to Streamlit's theme (.streamlit/config.toml).  Setting
   one globally also overrides Streamlit's icon font, which then renders the
   ligature names ("keyboard_arrow_right") as literal text. */

h1, h2, h3, h4 {{
  letter-spacing: -0.015em;
  color: {TEXT};
}}

/* ---- sidebar ---------------------------------------------------------- */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, {SIDEBAR} 0%, #081020 100%);
  border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .block-container {{
  padding-top: 1.2rem;
}}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: rgba(14, 26, 43, 0.6);
}}

/* ---- bordered containers as cards ------------------------------------- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  border-color: {BORDER} !important;
  border-radius: 16px;
  background: linear-gradient(172deg, {PANEL} 0%, #0B1524 100%);
}}

/* ---- tabs ------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {{
  gap: 4px;
  border-bottom: 1px solid {BORDER};
  padding-bottom: 2px;
}}
.stTabs [data-baseweb="tab"] {{
  background: transparent;
  color: {MUTED};
  border-radius: 10px 10px 0 0;
  padding: 8px 16px;
  font-size: 0.9rem;
  font-weight: 600;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: {TEXT}; }}
.stTabs [aria-selected="true"] {{
  color: {TEAL};
  background: rgba(47, 214, 192, 0.06);
  box-shadow: inset 0 -2px 0 0 {TEAL};
}}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: transparent; }}

/* ---- controls --------------------------------------------------------- */
div.stButton > button, div.stDownloadButton > button {{
  border-radius: 10px;
  border: 1px solid {BORDER};
  background: {PANEL_ALT};
  color: {TEXT};
  font-weight: 550;
  transition: border-color 120ms ease, background 120ms ease;
}}
div.stButton > button:hover {{
  border-color: {TEAL};
  background: #14273D;
  color: {TEXT};
}}
div.stButton > button[kind="primary"] {{
  background: linear-gradient(180deg, #17A899 0%, #0F8478 100%);
  border-color: #17A899;
  color: #04121A;
}}
div.stButton > button[kind="primary"]:hover {{ background: #1FC0AE; border-color: #1FC0AE; }}

[data-testid="stWidgetLabel"] p {{
  font-size: 0.78rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: {MUTED};
}}

div[data-baseweb="select"] > div,
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {{
  background: #0B1626;
  border-color: {BORDER};
  color: {TEXT};
}}

[data-testid="stChatInput"] {{
  background: {PANEL};
  border: 1px solid {BORDER};
  border-radius: 14px;
}}
[data-testid="stChatInput"] textarea {{ color: {TEXT}; }}

[data-testid="stChatMessage"] {{
  background: {PANEL};
  border: 1px solid {BORDER};
  border-radius: 14px;
  padding: 12px 14px;
}}

div[data-testid="stExpander"] details {{
  border: 1px solid {BORDER};
  border-radius: 12px;
  background: #0B1626;
}}
div[data-testid="stExpander"] summary {{
  color: {MUTED};
  font-size: 0.85rem;
  font-weight: 600;
}}

hr {{ border-color: {BORDER}; }}

/* ---- custom components ------------------------------------------------ */
.ts-brand {{
  font-size: 1.28rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: {TEXT};
}}
.ts-brand-accent {{ color: {TEAL}; }}
.ts-brand-sub {{
  font-size: 0.74rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: {DIM};
  margin-top: 2px;
}}

.ts-eyebrow {{
  font-size: 0.7rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: {DIM};
  font-weight: 600;
}}

.ts-title {{
  font-size: 1.5rem;
  font-weight: 680;
  color: {TEXT};
  margin: 0;
  letter-spacing: -0.02em;
}}
.ts-subtitle {{
  font-size: 0.84rem;
  color: {MUTED};
  margin-top: 3px;
}}

.ts-chip {{
  display: inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 0.7rem;
  font-weight: 650;
  letter-spacing: 0.05em;
  border: 1px solid {BORDER};
  background: rgba(17, 32, 52, 0.8);
  color: {MUTED};
  margin: 0 6px 6px 0;
}}
.ts-chip.is-teal {{ color: {TEAL}; border-color: #1D4B47; background: rgba(47, 214, 192, 0.08); }}
.ts-chip.is-blue {{ color: {BLUE}; border-color: #1E4258; background: rgba(72, 180, 245, 0.08); }}
.ts-chip.is-amber {{ color: {AMBER}; border-color: #4B3A1B; background: rgba(240, 177, 60, 0.08); }}
.ts-chip.is-red {{ color: {RED}; border-color: #4E2427; background: rgba(242, 122, 122, 0.08); }}

.ts-kpi {{
  border: 1px solid {BORDER};
  border-radius: 15px;
  padding: 13px 16px 14px 16px;
  background: linear-gradient(160deg, {PANEL_ALT} 0%, #0B1524 100%);
  min-height: 104px;
}}
.ts-kpi-label {{
  font-size: 0.68rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: {DIM};
  font-weight: 650;
}}
.ts-kpi-value {{
  font-size: 1.85rem;
  font-weight: 660;
  line-height: 1.15;
  margin-top: 7px;
  color: {TEXT};
  font-variant-numeric: tabular-nums;
}}
.ts-kpi-value.is-alert {{ color: {RED}; }}
.ts-kpi-value.is-ok {{ color: {GREEN}; }}
.ts-kpi-value.is-neutral {{ color: {TEXT}; }}
.ts-kpi-value.is-small {{ font-size: 1.32rem; }}
.ts-kpi-meta {{
  font-size: 0.76rem;
  color: {DIM};
  margin-top: 5px;
  line-height: 1.35;
}}

.ts-banner {{
  border: 1px solid {BORDER};
  border-left: 3px solid {AMBER};
  border-radius: 12px;
  padding: 10px 14px;
  background: rgba(240, 177, 60, 0.05);
  color: {MUTED};
  font-size: 0.82rem;
  line-height: 1.45;
}}
.ts-banner.is-teal {{ border-left-color: {TEAL}; background: rgba(47, 214, 192, 0.05); }}
.ts-banner.is-red {{ border-left-color: {RED}; background: rgba(242, 122, 122, 0.06); }}

.ts-muted {{ color: {MUTED}; font-size: 0.82rem; line-height: 1.5; }}
.ts-dim {{ color: {DIM}; font-size: 0.76rem; }}
.ts-mono {{ font-family: "JetBrains Mono", "Cascadia Code", ui-monospace, monospace; font-size: 0.78rem; color: {MUTED}; }}

.ts-meter {{
  position: relative;
  height: 12px;
  border-radius: 999px;
  background: #0A1422;
  border: 1px solid {BORDER};
  overflow: hidden;
  margin: 10px 0 4px 0;
}}
.ts-meter-fill {{
  position: absolute;
  inset: 0 auto 0 0;
  border-radius: 999px;
  background: linear-gradient(90deg, {TEAL} 0%, {BLUE} 62%, {AMBER} 100%);
}}
.ts-meter-mark {{
  position: absolute;
  top: -3px;
  bottom: -3px;
  width: 2px;
  background: {TEXT};
  opacity: 0.85;
}}
.ts-meter-scale {{
  display: flex;
  justify-content: space-between;
  font-size: 0.68rem;
  color: {DIM};
}}

.ts-availability {{
  border: 1px dashed {BORDER};
  border-radius: 12px;
  padding: 12px 14px;
  background: rgba(10, 20, 34, 0.65);
}}
.ts-availability-title {{
  font-size: 0.82rem;
  font-weight: 650;
  color: {AMBER};
  margin-bottom: 4px;
}}
.ts-footer {{
  color: {DIM};
  font-size: 0.74rem;
  line-height: 1.5;
  border-top: 1px solid {BORDER};
  padding-top: 12px;
  margin-top: 18px;
}}
.ts-list {{ margin: 0; padding-left: 18px; color: {MUTED}; font-size: 0.82rem; line-height: 1.55; }}

.ts-kv {{
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 7px 0;
  border-bottom: 1px solid rgba(29, 46, 71, 0.55);
  font-size: 0.84rem;
}}
.ts-kv:last-child {{ border-bottom: none; }}
.ts-kv-label {{ color: {MUTED}; }}
.ts-kv-value {{ color: {TEXT}; font-variant-numeric: tabular-nums; text-align: right; }}

code {{ color: {TEAL}; background: rgba(47, 214, 192, 0.08); border-radius: 4px; padding: 0 4px; font-size: 0.82em; }}
</style>
"""

__all__ = [
    "CSS",
    "NAVY",
    "SIDEBAR",
    "PANEL",
    "PANEL_ALT",
    "BORDER",
    "TEXT",
    "MUTED",
    "DIM",
    "TEAL",
    "BLUE",
    "AMBER",
    "RED",
    "GREEN",
    "GRID",
]
