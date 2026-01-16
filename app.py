import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

states_districts = {
    "Assam": ["Guwahati", "Dibrugarh", "Silchar", "Jorhat"],
    "Bihar": ["Patna", "Gaya", "Purnia", "Bhagalpur"],
    "Uttar Pradesh": ["Lucknow", "Varanasi", "Prayagraj", "Kanpur"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik"],
    "West Bengal": ["Kolkata", "Siliguri", "Durgapur", "Asansol"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Puri", "Balasore"],
    "Kerala": ["Kochi", "Trivandrum", "Kozhikode", "Thrissur"]
}

rows = []
start_date = datetime(2023, 6, 1)

for state, districts in states_districts.items():
    for district in districts:
        for day in range(75):  # 7 states × 4 districts × 75 days ≈ 2100 rows
            date = start_date + timedelta(days=day)
            rainfall = np.random.randint(50, 500)
            river_level = np.random.randint(100, 800)
            latitude = round(np.random.uniform(8, 28), 4)
            longitude = round(np.random.uniform(72, 97), 4)

            rows.append([
                state,
                district,
                latitude,
                longitude,
                date.strftime("%Y-%m-%d"),
                rainfall,
                river_level
            ])

df = pd.DataFrame(rows, columns=[
    "state", "district", "latitude", "longitude",
    "date", "rainfall_mm", "river_level_mm"
])

df.to_csv("proper_india_flood_data.csv", index=False)
print("CSV created with rows:", len(df))
