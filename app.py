# flood_risk_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="🌧️ Flood Risk Dashboard", layout="wide")
st.title("🌧️ Flood Risk Prediction & Visualization Dashboard")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload CSV with columns: state, district, latitude, longitude, date, rainfall_mm, river_level_mm")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Full Dataset Preview")
    st.dataframe(df)

    # --- Calculate Risk ---
    def calculate_risk(row):
        if row['rainfall_mm'] > 100 or row['river_level_mm'] > 500:
            return "High"
        elif row['rainfall_mm'] > 50 or row['river_level_mm'] > 250:
            return "Medium"
        else:
            return "Low"

    df['risk'] = df.apply(calculate_risk, axis=1)
    st.subheader("Dataset with Predicted Flood Risk")
    st.dataframe(df)

    # --- Download CSV with risk ---
    csv_exp = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV with Flood Risk",
        data=csv_exp,
        file_name='flood_risk_full.csv',
        mime='text/csv'
    )

    # --- State-wise Aggregation for Charts ---
    state_avg = df.groupby('state')[['rainfall_mm','river_level_mm']].mean().reset_index()

    st.subheader("State-wise Average Rainfall & River Level")
    fig1 = px.bar(state_avg, x='state', y=['rainfall_mm','river_level_mm'],
                  barmode='group', title="🌧️ Average Rainfall & River Level per State")
    st.plotly_chart(fig1, use_container_width=True)

    # --- State-wise Risk Distribution (Stacked Bar) ---
    st.subheader("State-wise Flood Risk Distribution")
    state_risk = df.groupby(['state','risk']).size().unstack(fill_value=0)
    fig2 = px.bar(state_risk, x=state_risk.index, y=['Low','Medium','High'],
                  title="🌊 State-wise Risk Distribution", color_discrete_map={'Low':'green','Medium':'orange','High':'red'})
    st.plotly_chart(fig2, use_container_width=True)

    # --- Sunburst: State → District Hierarchy for Rainfall ---
    st.subheader("🌧️ Rainfall State → District Hierarchy")
    df_agg = df.groupby(['state','district']).agg({'rainfall_mm':'mean','river_level_mm':'mean'}).reset_index()
    fig3 = px.sunburst(df_agg, path=['state','district'], values='rainfall_mm',
                       color='rainfall_mm', color_continuous_scale='Blues',
                       title="Rainfall Hierarchy (State → District)")
    st.plotly_chart(fig3, use_container_width=True)

    # --- Sunburst: State → District Hierarchy for River Level ---
    st.subheader("🌊 River Level State → District Hierarchy")
    fig4 = px.sunburst(df_agg, path=['state','district'], values='river_level_mm',
                       color='river_level_mm', color_continuous_scale='Oranges',
                       title="River Level Hierarchy (State → District)")
    st.plotly_chart(fig4, use_container_width=True)

    # --- ML Model for Risk Prediction ---
    st.subheader("⚡ Predict Flood Risk for New Data")
    X = df[['rainfall_mm','river_level_mm']]
    le = LabelEncoder()
    y = le.fit_transform(df['risk'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    st.write(f"Random Forest Model Accuracy: {acc*100:.2f}%")

    # --- User Input Prediction ---
    new_rainfall = st.number_input("Enter Rainfall (mm) for Prediction", min_value=0)
    new_river_level = st.number_input("Enter River Level (mm) for Prediction", min_value=0)
    if st.button("Predict Flood Risk"):
        pred = model.predict([[new_rainfall,new_river_level]])
        pred_label = le.inverse_transform(pred)[0]
        st.success(f"Predicted Flood Risk: {pred_label}")

else:
    st.info("Please upload a CSV file to start predicting flood risk.")
