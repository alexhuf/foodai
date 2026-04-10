import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("🌌 FOODAI V5: INITIALIZING OMNISCIENT DATA EXTRACTION...")

# --- 1. LOAD & MERGE RAW DATA ---
try:
    df_super = pd.read_csv('Ultimate_Super_Dataset.csv')
    df_raw = pd.read_csv('android_food_entries.csv')
except FileNotFoundError as e:
    print(f"ERROR: Missing dataset: {e}")
    exit()

df_super['uuid'] = df_super['uuid'].astype(str)
df_raw['uuid'] = df_raw['uuid'].astype(str)

# Extract precise timestamp
df = pd.merge(df_super, df_raw[['uuid', 'clientTimeInserted']], on='uuid', how='left')
df['Exact_Time'] = pd.to_datetime(df['clientTimeInserted'], errors='coerce')
fallback = pd.to_datetime(df['date']) + pd.to_timedelta(df['timeSlot'] * 4, unit='h')
df['Exact_Time'] = df['Exact_Time'].fillna(fallback)
df['date'] = pd.to_datetime(df['date'])

# [BUG FIX]: Calculate Is_Weekend early so it exists for the Context Grouping
df['Is_Weekend'] = df['Exact_Time'].dt.dayofweek.isin([5, 6]).astype(int)

# --- 2. DAILY AGGREGATIONS FOR ECHO MATRIX ---
print("⚙️ Compiling Daily Baselines...")
daily = df.groupby('date').agg({
    'calories': 'sum', 'carbs_g': 'sum', 'protein_g': 'sum', 'fat_g': 'sum',
    'calorieBudget': 'first', 'steps': 'first', 'weight_lbs': 'first'
}).reset_index()

daily['Daily_Deficit'] = daily['calorieBudget'] - daily['calories']
# Energy Flux = Total Intake / Total Output (Approximated by steps)
daily['Energy_Flux'] = daily['calories'] / (daily['steps'] + 1) # +1 avoids division by zero

# --- 3. THE MICRO-TO-MACRO TIME HORIZONS (The 1000D Generator) ---
print("⚙️ Generating 9-Tier Time Horizons (1 to 120 Days)...")
horizons = [1, 3, 5, 7, 14, 30, 60, 90, 120]
metrics = ['calories', 'carbs_g', 'protein_g', 'fat_g', 'steps', 'weight_lbs', 'Daily_Deficit', 'Energy_Flux']

for h in horizons:
    for m in metrics:
        # Calculate Rolling Averages
        daily[f'Avg_{h}D_{m}'] = daily[m].rolling(h, min_periods=1).mean()
        # Calculate Volatility (Standard Deviation) - Critical for spotting behavioral instability
        if h > 1:
            daily[f'Vol_{h}D_{m}'] = daily[m].rolling(h, min_periods=1).std().fillna(0)

cols_to_merge = [c for c in daily.columns if 'Avg_' in c or 'Vol_' in c]
df = pd.merge(df, daily[['date'] + cols_to_merge], on='date', how='left')

# --- 4. THE RECONSTITUTION ENGINE (Plate Rollup) ---
print("🍽️ Assembling Autoregressive Meal Sequences...")
grouped = df.groupby(['date', 'timeSlot'])

# Create a strict sequential list of ingredients for the Transformer Decoder
plates_df = grouped['Display_Name'].apply(lambda x: list(x)).reset_index(name='Sequence_Target')

plate_stats = grouped.agg({
    'calories': 'sum', 'carbs_g': 'sum', 'protein_g': 'sum', 'fat_g': 'sum',
    'Exact_Time': 'min'
}).reset_index()

# Pull all the 1000D horizon columns
context_cols = ['calorieBudget', 'steps', 'weight_lbs', 'Approx_High_Temp', 'Is_Weekend'] + cols_to_merge
plate_context = grouped[context_cols].first().reset_index()

plates_df = pd.merge(plates_df, plate_stats, on=['date', 'timeSlot'])
plates_df = pd.merge(plates_df, plate_context, on=['date', 'timeSlot'])
plates_df = plates_df.sort_values('Exact_Time').reset_index(drop=True)

# --- 5. CHRONOBIOLOGY & PSYCHOLOGICAL FRICTION ---
print("⏱️ Mapping Psychological Friction & Fasting Cascades...")

# Calculate Fasting Time (Hours since LAST plate)
plates_df['Fasting_Hours_Current'] = plates_df['Exact_Time'].diff().dt.total_seconds() / 3600
plates_df['Fasting_Hours_Current'] = plates_df['Fasting_Hours_Current'].fillna(12.0)

# Calculate the Rolling Average of Fasting Times (Are your fasts getting longer/shorter?)
plates_df['Fasting_Avg_30D'] = plates_df['Fasting_Hours_Current'].rolling(30, min_periods=1).mean()

# Intraday Velocity
plates_df['Cals_Consumed_Today_Prior'] = plates_df.groupby('date')['calories'].cumsum() - plates_df['calories']
plates_df['Remaining_Budget'] = plates_df['calorieBudget'] - plates_df['Cals_Consumed_Today_Prior']

# Ingredient Degradation (How complex is this meal?)
plates_df['Ingredient_Count'] = plates_df['Sequence_Target'].apply(len)
plates_df['Ingredient_Count_30D_Avg'] = plates_df['Ingredient_Count'].rolling(30, min_periods=1).mean()

# The "Guilt Delta" (Difference between yesterday's deficit and today's)
plates_df['Guilt_Delta_24H'] = plates_df['Avg_1D_Daily_Deficit'].diff().fillna(0)

# Clean up any anomalies
plates_df = plates_df[plates_df['calories'] > 0]

print("📦 Freezing V5 Omni-Matrix...")
# Convert target sequence to a string so it saves to CSV safely
plates_df['Sequence_Target_String'] = plates_df['Sequence_Target'].apply(lambda x: " | ".join(str(i) for i in x))
plates_df.drop(columns=['Sequence_Target']).to_csv('v5_omni_matrix.csv', index=False)

print(f"✅ V5 MATRIX COMPILED SUCCESSFULLY!")
print(f"Total Deep Biological Dimensions Extracted: {len(plates_df.columns)}")