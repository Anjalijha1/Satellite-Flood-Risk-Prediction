import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Flood Risk Prediction", layout="wide")

st.title("Flood Risk Detection & Prediction Dashboard")

st.write(
    "Upload flood-related CSV data. "
    "The system uses Machine Learning to analyze and predict flood risk."
)

# -----------------------
# FILE UPLOAD
# -----------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

required_cols = [
    "state", "rainfall_mm", "river_level_mm", "floodrisk"
]

for col in required_cols:
    if col not in df.columns:
        st.error(f"Missing required column: {col}")
        st.stop()

st.subheader("Uploaded Data")
st.dataframe(df, use_container_width=True)

# -----------------------
# LABEL ENCODING
# -----------------------
le = LabelEncoder()
df["risk_encoded"] = le.fit_transform(df["floodrisk"])

X = df[["rainfall_mm", "river_level_mm"]]
y = df["risk_encoded"]

# -----------------------
# ML MODEL
# -----------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)

# -----------------------
# PREDICTION
# -----------------------
df["predicted_encoded"] = model.predict(X)
df["predicted_flood_risk"] = le.inverse_transform(df["predicted_encoded"])

# -----------------------
# RESULT TABLE
# -----------------------
st.subheader("Flood Risk Prediction Result")
st.dataframe(
    df[
        [
            "state",
            "rainfall_mm",
            "river_level_mm",
            "floodrisk",
            "predicted_flood_risk"
        ]
    ],
    use_container_width=True
)

# -----------------------
# PIE CHART
# -----------------------
st.subheader("Flood Risk Distribution")

risk_counts = df["predicted_flood_risk"].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(
    risk_counts,
    labels=risk_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
st.pyplot(fig1)

# -----------------------
# STATE FILTER
# -----------------------
st.subheader("State-wise Analysis")

selected_state = st.selectbox(
    "Select State",
    df["state"].unique()
)

state_df = df[df["state"] == selected_state]
st.dataframe(state_df, use_container_width=True)

# -----------------------
# GRAPHS
# -----------------------
st.subheader("Rainfall vs River Level")

fig2, ax2 = plt.subplots()
ax2.scatter(df["rainfall_mm"], df["river_level_mm"])
ax2.set_xlabel("Rainfall (mm)")
ax2.set_ylabel("River Level (mm)")
st.pyplot(fig2)

st.subheader("Average Rainfall by State")

avg_rain = df.groupby("state")["rainfall_mm"].mean()

fig3, ax3 = plt.subplots()
avg_rain.plot(kind="bar", ax=ax3)
ax3.set_ylabel("Rainfall (mm)")
st.pyplot(fig3)

st.write(
    "Model used: Random Forest Classifier. "
    "This can be extended using satellite imagery and temporal data."
)
