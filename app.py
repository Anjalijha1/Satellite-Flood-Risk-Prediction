# flood_risk_app_full.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Page Config ---
st.set_page_config(page_title="🌧️ Flood Risk Prediction App", layout="wide")
st.title("🌧️ Flood Risk Prediction Dashboard")

# --- File Upload ---
st.sidebar.header("Upload CSV Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV with columns: state, district, latitude, longitude, date, rainfall_mm, river_level_mm")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Full Dataset Preview")
    st.dataframe(df)  # scrollable table, shows all rows

    # --- Risk Calculation (Heuristic Approach) ---
    def calculate_risk(row):
        if row['rainfall_mm'] > 100 or row['river_level_mm'] > 500:
            return "High"
        elif row['rainfall_mm'] > 50 or row['river_level_mm'] > 250:
            return "Medium"
        else:
            return "Low"

    df['risk'] = df.apply(calculate_risk, axis=1)

    st.subheader("Dataset with Flood Risk Column")
    st.dataframe(df)  # Shows all rows with risk

    # --- Download CSV with risk ---
    csv_exp = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV with Flood Risk",
        data=csv_exp,
        file_name='flood_risk_full.csv',
        mime='text/csv'
    )

    # --- Pie Chart of Risk Levels ---
    st.subheader("Flood Risk Distribution")
    risk_counts = df['risk'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=['red','orange','green'])
    ax1.set_title("Flood Risk Levels")
    st.pyplot(fig1)

    # --- Scatter Plot: Rainfall vs River Level colored by Risk ---
    st.subheader("Rainfall vs River Level by Risk")
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.scatterplot(
        data=df,
        x='rainfall_mm',
        y='river_level_mm',
        hue='risk',
        palette={'Low':'green','Medium':'orange','High':'red'},
        ax=ax2
    )
    ax2.set_title("Rainfall vs River Level")
    st.pyplot(fig2)

    # --- ML Model: Predict Risk from Rainfall and River Level ---
    st.subheader("Flood Risk Prediction Model (Random Forest)")

    X = df[['rainfall_mm', 'river_level_mm']]
    le = LabelEncoder()
    y = le.fit_transform(df['risk'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    st.write(f"Model Accuracy: {accuracy*100:.2f}%")

    # --- Predict Risk for New Input ---
    st.subheader("Predict Flood Risk for New Data")
    new_rainfall = st.number_input("Enter Rainfall (mm)", min_value=0)
    new_river_level = st.number_input("Enter River Level (mm)", min_value=0)
    if st.button("Predict Risk"):
        pred = model.predict([[new_rainfall, new_river_level]])
        pred_label = le.inverse_transform(pred)[0]
        st.success(f"Predicted Flood Risk: {pred_label}")

else:
    st.info("Please upload a CSV file to start predicting flood risk.")
