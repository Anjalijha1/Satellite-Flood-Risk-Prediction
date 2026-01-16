import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_csv('weather_state_district_no_floodrisk_4000_rows.csv')

# 2. Prepare Data for Machine Learning (K-Means)
# We use rainfall and river level to predict risk clusters
features = ['rainfall_mm', 'river_level_mm']
X = df[features]

# Scale features for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply K-Means Clustering (3 clusters for Low, Moderate, High)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 4. Map Clusters to Human-Readable Risk Labels
# We rank clusters by their average rainfall/river level to assign labels correctly
cluster_means = df.groupby('cluster')[features].mean().sum(axis=1).sort_values()
label_map = {
    cluster_means.index[0]: 'Low Risk',
    cluster_means.index[1]: 'Moderate Risk',
    cluster_means.index[2]: 'High Risk'
}
df['predicted_flood_risk'] = df['cluster'].map(label_map)
df.drop(columns=['cluster'], inplace=True) # Remove helper column

# 5. Save the updated dataset
df.to_csv('flood_risk_predictions_final.csv', index=False)

# 6. Visualization: Pie Chart
plt.figure(figsize=(8, 8))
risk_counts = df['predicted_flood_risk'].value_counts()
plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', 
        colors=['#99ff99','#66b3ff','#ff9999'], startangle=140, explode=(0.05, 0.05, 0.05))
plt.title('Total Distribution of Predicted Flood Risk')
plt.savefig('flood_risk_pie_chart.png')

# 7. Beautiful Table Output (First 10 Rows)
print("Updated Dataset with Prediction:")
print(df.head(10).to_markdown())
