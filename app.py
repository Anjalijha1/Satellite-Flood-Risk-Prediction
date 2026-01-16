import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_flood_app_logic(file_path):
    # 1. Load the User's Uploaded CSV
    df = pd.read_csv(file_path)
    
    # 2. Prepare Machine Learning Features
    # We use rainfall and river level for clustering
    features = ['rainfall_mm', 'river_level_mm']
    X = df[features]
    
    # Standardize data for accurate ML clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. K-Means Clustering (3 levels: Low, Moderate, High)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(X_scaled)
    
    # 4. Map Cluster IDs to Risk Labels Automatically
    # Sort clusters by severity (average values) to ensure 'High' is always high values
    severity_order = df.groupby('cluster_id')[features].mean().sum(axis=1).sort_values().index
    risk_labels = {severity_order[0]: 'Low Risk', 
                   severity_order[1]: 'Moderate Risk', 
                   severity_order[2]: 'High Risk'}
    
    df['predicted_flood_risk'] = df['cluster_id'].map(risk_labels)
    df_final = df.drop(columns=['cluster_id'])
    
    # 5. Visualizations
    # Pie Chart
    plt.figure(figsize=(10, 6))
    risk_counts = df_final['predicted_flood_risk'].value_counts()
    plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', 
            colors=['#2ecc71', '#f1c40f', '#e74c3c'], startangle=140, shadow=True)
    plt.title('Total Flood Risk Analysis')
    plt.savefig('app_pie_chart.png')
    
    # 6. Future Trend Prediction (Simple Linear Projection)
    # Taking top 5 High Risk areas for a future projection graph
    high_risk_sample = df_final[df_final['predicted_flood_risk'] == 'High Risk'].head(1)
    if not high_risk_sample.empty:
        base_val = high_risk_sample['rainfall_mm'].values[0]
        future_data = [base_val * (1 + (i * 0.15)) for i in range(7)]
        plt.figure(figsize=(10, 4))
        plt.plot(['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'], 
                 future_data, marker='o', color='red')
        plt.title('7-Day Rainfall Future Prediction (Trend)')
        plt.savefig('app_future_trend.png')

    # Save final CSV
    df_final.to_csv('app_output_predicted_data.csv', index=False)
    return df_final

# How to use:
# processed_data = run_flood_app_logic('your_uploaded_file.csv')
