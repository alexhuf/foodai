import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🧠 INITIALIZING V3 CONTEXTUAL MEMORY ENGINE...")

# 1. Load the Plate Dataset
df = pd.read_csv('V3_Deep_Plate_Dataset.csv')

# 2. Encode Strings to Numbers (KNN requires pure math)
categorical_cols = ['Day_of_Week', 'Month']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# 3. Define the 45+ Dimensional Feature Matrix
# We exclude the text labels (like Full_Meal_Plate) and the target macros for the specific meal
exclude_cols = ['date', 'timeSlot', 'Meal_Time', 'Full_Meal_Plate', 
                'Primary_Food_Item', 'Plate_Micro_Cuisine', 'Plate_Protein', 
                'Plate_Preparation', 'Plate_Archetype', 
                'calories', 'carbs_g', 'protein_g', 'fat_g']

feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].fillna(0)

# 4. Scale the Data (Crucial for Euclidean Distance)
print("⚖️ Scaling the 45-Dimensional Matrix...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Build the K-Nearest Neighbors Memory Engine
print("🌌 Mapping your history into hyperspace...")
# metric='minkowski' (p=2) is standard Euclidean distance
knn = NearestNeighbors(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_scaled)

# 6. Freeze the Architecture
print("📦 Freezing the V3 Brain...")
with open('v3_memory_engine.pkl', 'wb') as f:
    pickle.dump({
        'knn_model': knn,
        'scaler': scaler,
        'encoders': encoders,
        'feature_cols': feature_cols,
        # We save the dataframe so the app can look up the literal food strings
        'reference_data': df[['Full_Meal_Plate', 'Primary_Food_Item', 'Plate_Archetype', 'calories', 'carbs_g', 'protein_g', 'fat_g']]
    }, f)

print(f"✅ V3 TRAINING COMPLETE! Brain is tracking {len(feature_cols)} dimensions.")