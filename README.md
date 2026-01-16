# 🌧️ Flood Risk Prediction & Visualization Dashboard

This is an **interactive Python Streamlit dashboard** that predicts flood risk for districts in India (or any region) based on **rainfall** and **river level** data. It also provides **state-wise and district-wise visualizations** to help analyze and understand flood risk distribution.

---

## 🚀 Features

1. **Automatic Flood Risk Prediction**
   - Predicts **Low, Medium, High** risk for each district.
   - Uses **dynamic percentile-based thresholds** based on your dataset.
   - Optional **Random Forest ML model** for predicting flood risk for new input.

2. **State-wise Visualizations**
   - **Average Rainfall & River Level per state** (grouped bar chart).  
   - **Flood Risk Distribution** (stacked bar chart showing number of districts per risk category).  
   - **Interactive Sunburst Charts** for **State → District hierarchy**:
     - Rainfall hierarchy.
     - River level hierarchy.

3. **Interactive Prediction**
   - Users can enter **rainfall & river level values** to predict flood risk for new cases.

4. **Data Export**
   - Download the **full dataset with predicted risk** as a CSV.

---

## 📊 How It Works

1. **Upload CSV**
   - The CSV should contain the following columns:
     ```
     state, district, latitude, longitude, date, rainfall_mm, river_level_mm
     ```

2. **Risk Calculation**
   - Each row (district) is assigned a **risk level**:
     - **High** → top 33% of rainfall or river level.  
     - **Medium** → middle 33%.  
     - **Low** → bottom 33%.  

3. **Charts**
   - **State-wise Flood Risk Distribution**:  
     - Y-axis = number of districts.  
     - Colors = risk level (Green=Low, Orange=Medium, Red=High).  
   - **Sunburst Charts**:  
     - Interactive hierarchy: click on state → see its districts.  
     - Color-coded by rainfall or river level.  

4. **ML Prediction**
   - Random Forest model is trained on your data (`rainfall_mm` & `river_level_mm`) to predict risk.  
   - You can enter new values for rainfall & river level to get predicted risk.

---

## 🛠️ Requirements

- Python 3.8+
- Libraries:
