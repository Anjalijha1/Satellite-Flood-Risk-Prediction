import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def process_flood_data(csv_file):
    # 1. Load Data
    df = pd.read_csv(csv_file)
    
    # 2. Machine Learning Logic (K-Means)
    # Using Rainfall and River Level to group regions into 3 Risk levels
    features = ['rainfall_mm', 'river_level_mm']
    X = df[features]
    
    # Standardize data for accurate clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Predict 3 Clusters (Low, Moderate, High)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Map clusters to Risk Labels automatically
    # Rank clusters by severity (sum of mean values)
    severity = df.groupby('cluster')[features].mean().sum(axis=1).sort_values()
    risk_map = {severity.index[0]: 'Low Risk', 
                severity.index[1]: 'Moderate Risk', 
                severity.index[2]: 'High Risk'}
    
    df['predicted_flood_risk'] = df['cluster'].map(risk_map)
    df_final = df.drop(columns=['cluster'])
    
    # 3. Generate Visuals
    # Pie Chart
    plt.figure(figsize=(8, 6))
    risk_counts = df_final['predicted_flood_risk'].value_counts()
    plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', 
            colors=['#2ecc71', '#f1c40f', '#e74c3c'], startangle=140, shadow=True)
    plt.title('Flood Risk Distribution')
    plt.savefig('app_pie_chart.png')
    
    # Return processed dataframe
    return df_final

# Example usage in your app:
# processed_df = process_flood_data('your_uploaded_file.csv')
# print(processed_df.head())
