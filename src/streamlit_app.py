import streamlit as st
import requests
import os
import pandas as pd
from datetime import datetime

# Configure Backend URL (Use container network hostname 'api' inside Docker)
BACKEND_URL = os.getenv("BACKEND_URL", "http://api:8000/predict")

st.set_page_config(
    page_title="AeroScrape - Flight Delay Predictor",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ AeroScrape: Flight Delay Predictor")
st.write("Enter the flight details below to evaluate the estimated delay in minutes.")

with st.form("prediction_form"):
    airline = st.selectbox(
        "Airline",
        ["کاسپین", "اطلس ایر", "آتا", "پويا", "ایران ایر", "ایران ایرتور", "معراج", "ماهان"]
    )
    
    airport = st.selectbox(
        "Airport",
        ["فرودگاه مهرآباد", "فرودگاه امام خمینی", "فرودگاه مشهد", "فرودگاه شیراز"]
    )
    
    destination_or_origin = st.text_input("Destination or Origin City", value="مشهد")
    aircraft = st.text_input("Aircraft Model", value="MD83")
    
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
        "scheduled_datetime": scheduled_datetime
    }
    
    try:
        with st.spinner("Calculating delay prediction and feature impact (SHAP)..."):
            response = requests.post(BACKEND_URL, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            delay = result["predicted_delay_minutes"]
            is_delayed = result["is_delayed"]
            explainability = result.get("explainability", [])
            
            st.success("Analysis Complete!")
            
            # 1. Main prediction display
            if is_delayed:
                st.error(f"⚠️ Predicted Delay: {delay:.2f} minutes")
            else:
                st.success(f"✅ On Time! Predicted Delay: {delay:.2f} minutes")
                
            # 2. SHAP Explainability Display
            if explainability:
                st.subheader("📊 Why did the model make this prediction? (SHAP)")
                st.write("Below are the top factors that impacted this specific estimation:")
                
                pos_factors = [f for f in explainability if f["contribution"] > 0]
                neg_factors = [f for f in explainability if f["contribution"] < 0]
                
                exp_col1, exp_col2 = st.columns(2)
                
                with exp_col1:
                    st.markdown("**🔴 Factors Increasing Delay**")
                    if pos_factors:
                        for item in pos_factors[:5]:  # Show top 5
                            st.write(f"- **{item['feature']}**: +{item['contribution']:.1f} mins")
                    else:
                        st.write("_No major delaying factors detected_")
                        
                with exp_col2:
                    st.markdown("**🟢 Factors Reducing Delay**")
                    if neg_factors:
                        for item in neg_factors[:5]:  # Show top 5
                            st.write(f"- **{item['feature']}**: {item['contribution']:.1f} mins")
                    else:
                        st.write("_No major reducing factors detected_")

                # Optional: Render as a quick impact bar chart
                st.markdown("---")
                st.markdown("**Feature Impact Bar Chart**")
                chart_data = pd.DataFrame(explainability).rename(
                    columns={"feature": "Factor", "contribution": "Minutes Contribution"}
                ).set_index("Factor")
                st.bar_chart(chart_data)
                
        else:
            st.error(f"Backend API returned an error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to the Backend API at {BACKEND_URL}. Check if your Docker container stack is fully running.")