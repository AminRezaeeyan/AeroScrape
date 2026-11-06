"""Streamlit demo of the delay prediction API.

    streamlit run src/aeroscrape/demo/streamlit_app.py   # from the repo root, for .streamlit/config.toml

The dropdowns offer the values the served model was trained on (GET /v1/model); a value it
never saw would be ignored by the model, so it isn't offered.
"""

import html
import os
from datetime import date, datetime, time, timedelta
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://api:8000/v1/predict")
API_BASE = BACKEND_URL.rsplit("/v1/", 1)[0]
API_KEY = os.getenv("AEROSCRAPE_API_KEY")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}
# Links in the sidebar open in the visitor's browser, so they use the published ports
LINKS = {
    "Grafana": os.getenv("GRAFANA_URL", "http://localhost:3000"),
    "MLflow": os.getenv("MLFLOW_URL", "http://localhost:5000"),
    "Airflow": os.getenv("AIRFLOW_URL", "http://localhost:8080"),
    "API docs": os.getenv("API_DOCS_URL", "http://localhost:8000/docs"),
}

FEATURE_NAMES = {
    "airline": "Airline",
    "airport": "Airport",
    "destination_or_origin": "Destination",
    "aircraft": "Aircraft",
    "Scheduled_Hour_of_Day": "Departure hour",
    "scheduled_minute": "Departure minute",
    "scheduled_day_of_week": "Weekday",
    "is_weekend": "Weekend",
    "scheduled_time_of_day": "Time of day",
    "has_inbound": "Inbound flight known",
    "scheduled_turnaround_min": "Scheduled turnaround (min)",
    "nth_flight_of_day_for_aircraft": "Aircraft's flight of the day",
    "airline_delay_rate_7d": "Airline delay rate, last 7 days",
    "route_median_delay_7d": "Route median delay, last 7 days",
    "prev_departure_delay_min": "Aircraft's previous delay (min)",
    "inbound_delay_min": "Inbound delay (min)",
    "inbound_overdue_min": "Inbound overdue (min)",
}
FALLBACK_CATEGORIES = {
    "airline": ["ماهان", "ایران ایر", "آتا", "کاسپین", "وارش", "قشم ایر"],
    "airport": ["فرودگاه مهرآباد", "فرودگاه مشهد", "فرودگاه شیراز"],
    "destination_or_origin": ["مشهد", "تهران", "کیش", "شیراز"],
    "aircraft": ["737", "A320", "MD83", "F100"],
}

CSS = """
<style>
.block-container {padding-top: 3.2rem; max-width: 1120px;}
header[data-testid="stHeader"] {background: transparent;}
.hero {background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%); color: #f8fafc;
       border-radius: 1.1rem; padding: 1.6rem 1.9rem; margin-bottom: 1.4rem; position: relative; overflow: hidden;}
.hero:after {content: "✈"; position: absolute; right: 1.6rem; top: 0.2rem; font-size: 6.5rem; opacity: 0.12;
             transform: rotate(-12deg);}
.hero h1 {color: #fff; font-size: 2rem; font-weight: 700; margin: 0 0 0.2rem 0; padding: 0;}
.hero p {color: #cbd5e1; margin: 0; font-size: 1.02rem;}
.chips {margin-top: 0.9rem; display: flex; gap: 0.45rem; flex-wrap: wrap;}
.chip {background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.18); color: #e2e8f0;
       border-radius: 999px; padding: 0.18rem 0.75rem; font-size: 0.82rem;}
.card {background: #fff; border: 1px solid #e2e8f0; border-radius: 1rem; padding: 1.1rem 1.25rem;
       box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 8px 24px rgba(15,23,42,0.05);}
.kpi {min-height: 158px;}
.card .label {color: #64748b; font-size: 0.85rem; margin-bottom: 0.3rem;}
.card .value {font-size: 2.1rem; font-weight: 700; line-height: 1.15;}
.card .sub {color: #64748b; font-size: 0.82rem; margin-top: 0.35rem;}
.meter {height: 8px; background: #eef2f7; border-radius: 999px; margin-top: 0.7rem; overflow: hidden;}
.meter > div {height: 100%; border-radius: 999px;}
.verdict {border-radius: 0.9rem; padding: 0.85rem 1.1rem; margin: 1rem 0 0.4rem 0; font-weight: 500;}
.range {position: relative; height: 54px; margin: 0.6rem 0.4rem 0.2rem 0.4rem;}
.range .track {position: absolute; top: 22px; left: 0; right: 0; height: 10px; background: #eef2f7; border-radius: 999px;}
.range .band {position: absolute; top: 22px; height: 10px; border-radius: 999px;
              background: linear-gradient(90deg, #93c5fd, #2563eb);}
.range .mid {position: absolute; top: 15px; width: 4px; height: 24px; background: #0f172a; border-radius: 2px;}
.range .tick {position: absolute; top: 16px; width: 2px; height: 22px; background: #f59e0b;}
.range .lbl {position: absolute; top: 40px; font-size: 0.75rem; color: #64748b; transform: translateX(-50%); white-space: nowrap;}
.range .toplbl {position: absolute; top: -4px; font-size: 0.75rem; color: #0f172a; font-weight: 600;
                transform: translateX(-50%); white-space: nowrap;}
.section {font-weight: 600; font-size: 1.1rem; margin: 1.4rem 0 0.2rem 0;}
.muted {color: #64748b; font-size: 0.85rem;}
.rtl {direction: rtl; unicode-bidi: plaintext;}
section[data-testid="stSidebar"] a {color: #93c5fd !important;}
</style>
"""


@st.cache_data(ttl=60, show_spinner=False)
def model_info() -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_BASE}/v1/model", headers=HEADERS, timeout=5)
        return response.json() if response.ok else None
    except requests.RequestException:
        return None


def risk_color(p: float) -> str:
    return "#16a34a" if p < 0.35 else "#f59e0b" if p < 0.6 else "#dc2626"


def card(label: str, value: str, sub: str = "", meter: tuple[float, str] | None = None) -> str:
    bar = (
        f'<div class="meter"><div style="width:{meter[0] * 100:.0f}%;background:{meter[1]}"></div></div>'
        if meter
        else ""
    )
    return (
        f'<div class="card kpi"><div class="label">{label}</div><div class="value">{value}</div>{bar}'
        f'<div class="sub">{sub}</div></div>'
    )


def range_bar(p10: float, p50: float, p90: float) -> str:
    """The 10-90% delay range on a minutes axis, with the median and the 15-minute line."""
    lo_axis = min(0.0, p10)
    hi_axis = max(60.0, p90 * 1.1)

    def pos(minutes: float) -> float:
        return (minutes - lo_axis) / (hi_axis - lo_axis) * 100

    return f"""
<div class="range">
  <div class="track"></div>
  <div class="band" style="left:{pos(p10):.1f}%; width:{pos(p90) - pos(p10):.1f}%"></div>
  <div class="tick" style="left:{pos(15):.1f}%"></div>
  <div class="mid" style="left:calc({pos(p50):.1f}% - 2px)"></div>
  <div class="toplbl" style="left:{pos(p50):.1f}%">{p50:.0f} min</div>
  <div class="lbl" style="left:{pos(p10):.1f}%">{p10:.0f}</div>
  <div class="lbl" style="left:{pos(15):.1f}%; color:#b45309">15</div>
  <div class="lbl" style="left:{pos(p90):.1f}%">{p90:.0f} min</div>
</div>"""


def display_value(feature: str, value: str) -> str:
    """Rates as percentages and other numbers rounded; text as it is."""
    try:
        number = float(value)
    except ValueError:
        return value
    if feature.endswith("_rate_7d"):
        return f"{number:.0%}"
    return f"{number:.0f}" if number == int(number) else f"{number:.1f}"


def factors_chart(factors: list[dict[str, Any]]) -> alt.Chart:
    df = pd.DataFrame(factors)
    df["label"] = [
        f"{FEATURE_NAMES.get(f, f)}: {display_value(f, str(v))}"
        for f, v in zip(df["feature"], df["value"], strict=True)
    ]
    df["effect"] = ["raises the chance" if c > 0 else "lowers the chance" for c in df["contribution"]]
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X("contribution:Q", title="Contribution (log-odds)", axis=alt.Axis(grid=True, tickCount=5)),
            y=alt.Y("label:N", sort=alt.EncodingSortField("contribution", op="sum", order="descending"), title=None),
            color=alt.Color(
                "effect:N",
                scale=alt.Scale(domain=["raises the chance", "lowers the chance"], range=["#ef4444", "#22c55e"]),
                legend=None,
            ),
            tooltip=["label", alt.Tooltip("contribution:Q", format="+.3f")],
        )
        .properties(height=max(200, 44 * len(df)))
        .configure(font="Vazirmatn")
        .configure_axis(labelFontSize=13, titleFontSize=12, labelColor="#334155", domain=False, labelLimit=360)
        .configure_scale(bandPaddingInner=0.25)
        .configure_view(strokeWidth=0)
    )


def options(info: dict[str, Any] | None, column: str) -> list[str]:
    values = (info or {}).get("categories", {}).get(column) or FALLBACK_CATEGORIES[column]
    return list(values)


def pick(label: str, values: list[str], preferred: str, key: str, **kwargs: Any) -> str:
    index = values.index(preferred) if preferred in values else 0
    return st.selectbox(label, values, index=index, key=key, **kwargs)


# ---------------------------------------------------------------------------- page

st.set_page_config(page_title="AeroScrape · Delay forecast", page_icon="✈️", layout="wide")
st.html(CSS)
info = model_info()

with st.sidebar:
    st.markdown("### ✈️ AeroScrape")
    st.caption("Flight delay forecasts for Iranian domestic departures, learned from scraped flight boards.")
    if info:
        st.markdown(
            f"**Model** version {info['model_version']}  \n"
            f"**Variant** {info['variant'].replace('_', ' ')}  \n"
            f"**Flag threshold** {info['decision_threshold']:.0%}"
        )
        if info.get("trained_on"):
            start, end = (str(t)[:10] for t in info["trained_on"])
            st.caption(f"Trained on flights from {start} to {end}.")
    else:
        st.warning("The API is not reachable; predictions are unavailable.")
    st.divider()
    st.markdown("**The rest of the stack**")
    st.markdown("  \n".join(f"[{name} ↗]({url})" for name, url in LINKS.items()))
    st.divider()
    st.caption("The range covers about 8 in 10 flights. Explanations are SHAP values of the delay classifier.")

chips = (
    [f"model v{info['model_version']}", info["variant"].replace("_", " "), "calibrated probability", "10–90% range"]
    if info
    else ["API unavailable"]
)
st.html(
    '<div class="hero"><h1>Will my flight be delayed?</h1>'
    "<p>Chance of a delay over 15 minutes, the likely delay, and what drives it.</p>"
    f'<div class="chips">{"".join(f"<span class=chip>{html.escape(c)}</span>" for c in chips)}</div></div>'
)

form_col, result_col = st.columns([5, 7], gap="large")

with form_col, st.form("prediction_form", border=True):
    st.markdown("**Flight**")
    a, b = st.columns(2)
    with a:
        airline = pick("Airline", options(info, "airline"), "ماهان", "airline")
    with b:
        aircraft = pick("Aircraft", options(info, "aircraft"), "MD83", "aircraft")
    airport = pick("From", options(info, "airport"), "فرودگاه مهرآباد", "airport")
    destination = pick("To", options(info, "destination_or_origin"), "مشهد", "destination")
    c, d = st.columns(2)
    tomorrow = date.today() + timedelta(days=1)
    with c:
        scheduled_date = st.date_input("Date", value=tomorrow)
    with d:
        scheduled_time = st.time_input("Departure", value=time(8, 5), step=300)
    with st.expander("Match a scraped flight (optional)"):
        st.caption("With these, the aircraft's earlier flights that day are used.")
        flight_number = st.text_input("Flight number", placeholder="e.g. IRM1036")
        register = st.text_input("Registration", placeholder="e.g. EPMOH")
    submitted = st.form_submit_button("Predict delay", type="primary", use_container_width=True)

with result_col:
    if not submitted:
        st.html(
            '<div class="card" style="text-align:center;padding:3.2rem 1.5rem">'
            '<div style="font-size:2.6rem">🛫</div><div style="font-weight:600;font-size:1.15rem;margin-top:.4rem">'
            "Pick a flight and press Predict</div><div class=muted style=margin-top:.4rem>"
            "You'll get the chance of a delay, the likely range in minutes and the factors behind it.</div></div>"
        )
    else:
        payload = {
            "airline": airline,
            "airport": airport,
            "destination_or_origin": destination,
            "aircraft": aircraft,
            "scheduled_datetime": datetime.combine(scheduled_date, scheduled_time).strftime("%Y-%m-%d %H:%M:%S"),
            "flight_number": flight_number or None,
            "register": register or None,
        }
        try:
            with st.spinner("Predicting…"):
                response = requests.post(
                    BACKEND_URL, params={"explain": "true"}, json=payload, headers=HEADERS, timeout=30
                )
        except requests.RequestException as e:
            st.error(f"Could not reach the API at {BACKEND_URL}: {e}")
            st.stop()
        if not response.ok:
            st.error(f"The API returned {response.status_code}: {response.text}")
            st.stop()

        result = response.json()
        p = result["p_delay_over_15"]
        delay = result["delay_minutes"]
        color = risk_color(p)
        k1, k2, k3 = st.columns(3)
        k1.html(
            card(
                "Delay over 15 min",
                f'<span style="color:{color}">{p:.0%}</span>',
                "calibrated chance",
                (p, color),
            )
        )
        k2.html(card("Likely delay", f"{delay['p50']:.0f}<span style=font-size:1rem> min</span>", "median prediction"))
        k3.html(
            card(
                "Likely range",
                f"{delay['p10']:.0f}–{delay['p90']:.0f}<span style=font-size:1rem> min</span>",
                "8 in 10 flights land inside",
            )
        )

        if result["likely_delayed"]:
            verdict = (
                "#fef2f2",
                "#991b1b",
                f"Likely delayed. The chance is above the {result['decision_threshold']:.0%} threshold, where about 3 in 4 flagged flights are late.",
            )
        elif p >= 0.35:
            verdict = (
                "#fffbeb",
                "#92400e",
                "Could go either way. Below the threshold for flagging a delay, but not a safe bet.",
            )
        else:
            verdict = ("#f0fdf4", "#166534", "Likely on time, or close to it.")
        st.html(f'<div class="verdict" style="background:{verdict[0]};color:{verdict[1]}">{verdict[2]}</div>')
        st.html(
            '<div class="card"><div class="label">Delay range, minutes after the scheduled departure '
            '<span style="color:#b45309">· orange line: 15 minutes</span></div>'
            f"{range_bar(delay['p10'], delay['p50'], delay['p90'])}</div>"
        )

        factors = result.get("top_factors") or []
        if factors:
            st.html('<div class="section">What drives this prediction</div>')
            st.caption("Red bars raise the chance of a delay, green bars lower it.")
            st.altair_chart(factors_chart(factors), use_container_width=True)

        notes = result.get("warnings", [])
        if notes:
            with st.expander(f"Notes on the input ({len(notes)})"):
                for note in notes:
                    st.markdown(f"- {note}")
        st.caption(f"Model version {result['model_version']} · {payload['scheduled_datetime']}")
