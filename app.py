import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Flood Risk Detection", layout="wide")

st.title("Satellite-based Flood Risk Detection System")

st.write(
    "This dashboard analyzes rainfall, river level, and elevation data "
    "to estimate flood risk in different regions."
)

# -------------------------------
# Sample Flood Risk Dataset
# -------------------------------
data = {
    "Region": ["Assam", "Bihar", "Odisha", "Uttar Pradesh", "Maharashtra"],
    "Rainfall (mm)": [320, 280, 260, 300, 150],
    "River Level (m)": [6.8, 6.2, 5.5, 6.0, 3.2],
    "Elevation (m)": [90, 85, 120, 100, 350]
}

df = pd.DataFrame(data)

# Flood risk logic
def classify_risk(rainfall, river):
    if rainfall > 300 and river > 6.5:
        return "High"
    elif rainfall > 200 and river > 5.5:
        return "Medium"
    else:
        return "Low"

df["Flood Risk"] = df.apply(
    lambda x: classify_risk(x["Rainfall (mm)"], x["River Level (m)"]),
    axis=1
)

# -------------------------------
# Display Table
# -------------------------------
st.subheader("Flood Risk Data Table")
st.dataframe(df, use_container_width=True)

# -------------------------------
# Charts Section
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Rainfall by Region")
    fig1, ax1 = plt.subplots()
    ax1.bar(df["Region"], df["Rainfall (mm)"])
    ax1.set_ylabel("Rainfall (mm)")
    ax1.set_xlabel("Region")
    st.pyplot(fig1)

with col2:
    st.subheader("River Level by Region")
    fig2, ax2 = plt.subplots()
    ax2.plot(df["Region"], df["River Level (m)"], marker="o")
    ax2.set_ylabel("River Level (m)")
    ax2.set_xlabel("Region")
    st.pyplot(fig2)

# -------------------------------
# Pie Chart
# -------------------------------
st.subheader("Flood Risk Distribution")

risk_counts = df["Flood Risk"].value_counts()
fig3, ax3 = plt.subplots()
ax3.pie(risk_counts, labels=risk_counts.index, autopct="%1.1f%%")
st.pyplot(fig3)

# -------------------------------
# Satellite Image Section
# -------------------------------
st.subheader("Satellite Observation (Example)")

st.image(
    "https://upload.wikimedia.org/wikipedia/commons/6/6b/Flood_satellite_image.jpg",
    caption="Satellite Image Showing Flooded Area",
    use_column_width=True
)

# -------------------------------
# Conclusion
# -------------------------------
st.subheader("Conclusion")
st.write(
    "Regions marked as **High Risk** require immediate monitoring and disaster preparedness. "
    "This system can be extended using real satellite and rainfall APIs."
)
