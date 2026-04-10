import pandas as pd
import numpy as np

print("🔍 V8 REWARD EXTRACTION: INITIATING INVERSE REINFORCEMENT LEARNING...")

# 1. Load the Omni-Matrix
try:
    df = pd.read_csv('v5_omni_matrix.csv')
    df['date'] = pd.to_datetime(df['date'])
except FileNotFoundError:
    print("ERROR: v5_omni_matrix.csv not found.")
    exit()

# 2. Collapse to Daily Totals
daily = df.groupby('date').agg({
    'calories': 'sum',
    'protein_g': 'sum',
    'fat_g': 'sum',
    'carbs_g': 'sum',
    'steps': 'first',
    'weight_lbs': 'first',
    'Fasting_Hours_Current': 'mean', 
    'Avg_1D_Energy_Flux': 'mean' # [BUG FIX] Changed to match the Omni-Matrix column name
}).reset_index()

daily = daily.sort_values('date').reset_index(drop=True)

# 3. Calculate Rolling 7-Day Metrics (The Week in Question)
print("⚙️ Scanning Rolling 7-Day Timelines...")
daily['7D_Total_Cals'] = daily['calories'].rolling(7).sum()
daily['7D_Avg_Protein'] = daily['protein_g'].rolling(7).mean()
daily['7D_Avg_Fat'] = daily['fat_g'].rolling(7).mean()
daily['7D_Avg_Carbs'] = daily['carbs_g'].rolling(7).mean()
daily['7D_Avg_Steps'] = daily['steps'].rolling(7).mean()
daily['7D_Avg_Fasting'] = daily['Fasting_Hours_Current'].rolling(7).mean()

# 4. Calculate the Future Consequence (The Next 7 Days)
# We look 7 days INTO THE FUTURE to see if the weight increased
daily['Future_Weight_7D_From_Now'] = daily['weight_lbs'].shift(-7)
daily['Subsequent_Weight_Delta'] = daily['Future_Weight_7D_From_Now'] - daily['weight_lbs']

# Clean NaN values at the ends of the dataset
daily = daily.dropna()

# 5. Isolate the "Golden Weeks"
print("⚖️ Isolating High-Intake, Zero-Rebound Anomalies...")
# Condition 1: Weight Delta must be 0 or negative (No rebound)
no_rebound = daily[daily['Subsequent_Weight_Delta'] <= 0.0]

# Condition 2: Of those weeks, find the top 10% most "Glutinous" (Highest Calories)
calorie_threshold = no_rebound['7D_Total_Cals'].quantile(0.90)
golden_weeks = no_rebound[no_rebound['7D_Total_Cals'] >= calorie_threshold]

if golden_weeks.empty:
    print("⚠️ No weeks found matching this strict criteria. Lowering calorie threshold...")
    calorie_threshold = no_rebound['7D_Total_Cals'].quantile(0.75)
    golden_weeks = no_rebound[no_rebound['7D_Total_Cals'] >= calorie_threshold]

# 6. Extract the Biological Fingerprint
print("\n==================================================")
print("🏆 THE GOLDEN BIOLOGICAL FINGERPRINT EXTRACTED 🏆")
print("==================================================")
print(f"Total 'Perfect Weeks' Found: {len(golden_weeks)}")
print(f"Average Caloric Intake (7-Day): {golden_weeks['7D_Total_Cals'].mean():.0f} kcal")
print(f"Average Subsequent Weight Change: {golden_weeks['Subsequent_Weight_Delta'].mean():.2f} lbs\n")

print("--- The Target V8 Reward Parameters ---")
print(f"Target Daily Protein: {golden_weeks['7D_Avg_Protein'].mean():.0f}g")
print(f"Target Daily Fat:     {golden_weeks['7D_Avg_Fat'].mean():.0f}g")
print(f"Target Daily Carbs:   {golden_weeks['7D_Avg_Carbs'].mean():.0f}g")
print(f"Target Fasting Avg:   {golden_weeks['7D_Avg_Fasting'].mean():.1f} hours")
print(f"Target Daily Steps:   {golden_weeks['7D_Avg_Steps'].mean():.0f}")
print("==================================================\n")