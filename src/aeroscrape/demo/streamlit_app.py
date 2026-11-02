import os
from datetime import datetime

import requests
import streamlit as st

# Configure Backend URL (Use container network hostname 'api' inside Docker)
BACKEND_URL = os.getenv("BACKEND_URL", "http://api:8000/predict")

st.set_page_config(page_title="AeroScrape - Flight Delay Predictor", layout="centered")

st.title("AeroScrape: Flight Delay Predictor")
st.write("Enter the flight details below to evaluate the estimated delay in minutes.")

with st.form("prediction_form"):
    airline = st.selectbox(
        "Airline", ["کاسپین", "اطلس ایر", "آتا", "پويا", "ایران ایر", "ایران ایرتور", "معراج", "ماهان"]
    )

    airport = st.selectbox("Airport", ["فرودگاه مهرآباد", "فرودگاه امام خمینی", "فرودگاه مشهد", "فرودگاه شیراز"])

    destination_or_origin = st.text_input("Destination or Origin City", value="مشهد")
    aircraft = st.text_input("Aircraft Model", value="MD83")
    flight_number = st.text_input(
        "Flight number (optional)", help="If the flight was scraped, its aircraft rotation is used, e.g. IRM1036"
    )
    register = st.text_input("Aircraft registration (optional)", help="e.g. EPMOH")

    # Date and Time Inputs
    col1, col2 = st.columns(2)
    with col1:
        scheduled_date = st.date_input("Scheduled Date", value=datetime.today())
    with col2:
        scheduled_time = st.time_input("Scheduled Time", value=datetime.now().time())

    submit_btn = st.form_submit_button("Predict Delay")

if submit_btn:
    # Combine date and time to format YYYY-MM-DD HH:MM:SS
    scheduled_datetime = datetime.combine(scheduled_date, scheduled_time).strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "airline": airline,
        "airport": airport,
        "destination_or_origin": destination_or_origin,
        "aircraft": aircraft,
        "flight_number": flight_number or None,
        "register": register or None,
        "scheduled_datetime": scheduled_datetime,
    }

    try:
        with st.spinner("Predicting..."):
            response = requests.post(BACKEND_URL, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            probability = result["p_delay_over_15"]
            delay = result["delay_minutes"]

            col1, col2 = st.columns(2)
            col1.metric("Chance of a delay over 15 min", f"{probability:.0%}")
            col2.metric("Likely delay", f"{delay['p50']:.0f} min", help="Median prediction")
            st.write(
                f"Likely range: **{delay['p10']:.0f} to {delay['p90']:.0f} minutes** "
                "(about 8 in 10 flights fall inside this range)."
            )
            if result["likely_delayed"]:
                st.warning(
                    f"Likely delayed: the probability is above the {result['decision_threshold']:.0%} threshold, "
                    "where about 3 in 4 flagged flights are delayed."
                )

            factors = result.get("top_factors", [])
            if factors:
                st.subheader("What drove this prediction")
                st.caption(
                    f"Contributions in {result['explanation_units']}. Positive values raise the chance of a delay."
                )
                up = [f for f in factors if f["contribution"] > 0]
                down = [f for f in factors if f["contribution"] < 0]
                left, right = st.columns(2)
                with left:
                    st.markdown("**Raise the chance of a delay**")
                    for item in up or []:
                        st.write(f"- {item['feature']} = {item['value']} ({item['contribution']:+.2f})")
                    if not up:
                        st.write("_None_")
                with right:
                    st.markdown("**Lower the chance of a delay**")
                    for item in down or []:
                        st.write(f"- {item['feature']} = {item['value']} ({item['contribution']:+.2f})")
                    if not down:
                        st.write("_None_")

            for warning in result.get("warnings", []):
                st.info(warning)
            st.caption(f"Model version {result.get('model_version')}")
        else:
            st.error(f"Backend API returned an error: {response.text}")

    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the backend API at {BACKEND_URL}: {e}")
