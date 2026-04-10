import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("🔄 INITIALIZING MAXIMUM DIMENSIONALITY RESTRUCTURING...")

try:
    df = pd.read_csv('Ultimate_Super_Dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
except FileNotFoundError:
    print("ERROR: Ultimate_Super_Dataset.csv not found!")
    exit()

# --- 1. INTRADAY CALORIE VELOCITY ---
print("⚙️ Calculating Intraday Velocity...")
# Sort chronologically to track the day exactly as you lived it
df = df.sort_values(['date', 'timeSlot'])
# Calculate how many calories you ate TODAY before this exact meal
df['Calories_Consumed_So_Far'] = df.groupby('date')['calories'].cumsum() - df['calories']
df['Remaining_Intraday_Budget'] = df['calorieBudget'] - df['Calories_Consumed_So_Far']

# --- 2. THE STARVATION TIMER ---
print("⚙️ Calculating Fasting Windows...")
# Create a dummy datetime combining date and timeslot to measure gaps
df['approx_datetime'] = df['date'] + pd.to_timedelta(df['timeSlot'] * 4, unit='h')
df['Hours_Since_Last_Meal'] = df['approx_datetime'].diff().dt.total_seconds() / 3600
df['Hours_Since_Last_Meal'] = df['Hours_Since_Last_Meal'].fillna(12) # Default overnight fast

# --- 3. ROLLING MACROS & ADHERENCE STREAKS ---
print("⚙️ Engineering Rolling Macros and Psychological Streaks...")
daily_stats = df.groupby('date').agg({
    'calories': 'sum',
    'carbs_g': 'sum',
    'protein_g': 'sum',
    'fat_g': 'sum',
    'calorieBudget': 'first',
    'steps': 'first'
}).reset_index()

daily_stats['Daily_Deficit'] = daily_stats['calorieBudget'] - daily_stats['calories']

# Calculate the Psychological Streak (How many consecutive days in a deficit?)
daily_stats['In_Deficit'] = (daily_stats['Daily_Deficit'] > 0).astype(int)
daily_stats['Deficit_Streak'] = daily_stats['In_Deficit'].groupby((daily_stats['In_Deficit'] != daily_stats['In_Deficit'].shift()).cumsum()).cumsum()

for window in [3, 5, 7, 30]:
    # Standard Rolling Averages
    daily_stats[f'Rolling_{window}D_Steps'] = daily_stats['steps'].rolling(window, min_periods=1).mean()
    daily_stats[f'Rolling_{window}D_Deficit'] = daily_stats['Daily_Deficit'].rolling(window, min_periods=1).mean()
    daily_stats[f'Rolling_{window}D_Cals'] = daily_stats['calories'].rolling(window, min_periods=1).mean()
    
    # NEW: Rolling Macros (The Carb-Starved Vector)
    daily_stats[f'Rolling_{window}D_Carbs'] = daily_stats['carbs_g'].rolling(window, min_periods=1).mean()
    daily_stats[f'Rolling_{window}D_Protein'] = daily_stats['protein_g'].rolling(window, min_periods=1).mean()
    daily_stats[f'Rolling_{window}D_Fat'] = daily_stats['fat_g'].rolling(window, min_periods=1).mean()

# Merge all the new daily metrics back into the main dataframe
cols_to_drop = [c for c in df.columns if 'Rolling' in c]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Merge daily stats back (dropping duplicates to avoid column collisions)
df = pd.merge(df, daily_stats.drop(columns=['calories', 'carbs_g', 'protein_g', 'fat_g', 'calorieBudget', 'steps', 'Daily_Deficit', 'In_Deficit']), on='date', how='left')

# Add Deep Calendar context
df['Week_of_Month'] = df['date'].dt.day.apply(lambda x: (x-1)//7 + 1)
df['Is_Weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
df.fillna(0, inplace=True)

# --- 4. THE PLATE ROLLUP ---
print("🍽️ Compiling ingredients into complete meals...")
grouped = df.groupby(['date', 'timeSlot', 'Meal_Time'])

# A) Combine items into one literal plate string (e.g. "Chicken + Rice + Beans")
meal_baskets = grouped['Display_Name'].apply(lambda x: " + ".join(str(i) for i in x)).reset_index(name='Full_Meal_Plate')

# B) Sum the macros for the entire plate
macros = grouped[['calories', 'carbs_g', 'protein_g', 'fat_g']].sum().reset_index()

# C) Extract the massive 45+ dimensional state vector
state_vars = [
    'Day_of_Week', 'Month', 'calorieBudget', 'steps', 'weight_lbs', 
    'Daylight_Hours', 'Approx_High_Temp', 'Is_Freezing', 'Is_Hot',
    'Week_of_Month', 'Is_Weekend', 
    'Calories_Consumed_So_Far', 'Remaining_Intraday_Budget', 'Hours_Since_Last_Meal',
    'Deficit_Streak'
] + [f'Rolling_{w}D_{m}' for w in [3, 5, 7, 30] for m in ['Steps', 'Deficit', 'Cals', 'Carbs', 'Protein', 'Fat']]

context = grouped[state_vars].first().reset_index()

# D) Extract the Primary Item and its NLP Data
def get_primary(g):
    # Sort by calories to find the "Main Dish"
    primary = g.sort_values('calories', ascending=False).iloc[0]
    return pd.Series({
        'Primary_Food_Item': primary['Display_Name'],
        'Plate_Micro_Cuisine': primary['Micro_Cuisine'],
        'Plate_Protein': primary['Protein_Cuts'],
        'Plate_Preparation': primary['Preparation'],
        'Plate_Archetype': primary['Meal_Archetype']
    })

primary_tags = grouped.apply(get_primary).reset_index()

# --- 5. MERGE EVERYTHING INTO V3 ---
print("📦 Compiling the ultimate V3 Matrix...")
v3_df = pd.merge(meal_baskets, macros, on=['date', 'timeSlot', 'Meal_Time'])
v3_df = pd.merge(v3_df, context, on=['date', 'timeSlot', 'Meal_Time'])
v3_df = pd.merge(v3_df, primary_tags, on=['date', 'timeSlot', 'Meal_Time'])

# Drop weird 0-calorie log entries (like plain water or API artifacts)
v3_df = v3_df[v3_df['calories'] > 0]

v3_df.to_csv('V3_Deep_Plate_Dataset.csv', index=False)
print("✅ SUCCESS! Dataset transformed into V3_Deep_Plate_Dataset.csv")
print(f"Total Unique Historical Plates Generated: {len(v3_df)}")
print(f"Total Dimensions per Plate: {len(v3_df.columns)}")