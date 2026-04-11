import pandas as pd

print("✂️ TRIMMING THE LEDGER TO THE 'ACTIVE ERA'...")

# Load the massive dataset
df = pd.read_csv("v15_master_telemetry.csv", index_col="datetime", parse_dates=True)

# Find the exact boundaries of your meal tracking
meal_rows = df[df['Is_Meal_Event'] == 1]
first_meal_date = meal_rows.index.min()
last_meal_date = meal_rows.index.max()

# Establish the 30-Day Warm-Up Buffer
trim_start = first_meal_date - pd.Timedelta(days=30)
trim_end = last_meal_date + pd.Timedelta(days=1) # Keep the rest of the final day

print(f"First recorded meal: {first_meal_date}")
print(f"Last recorded meal: {last_meal_date}")
print(f"Trimming dataset to window: {trim_start} -> {trim_end}")

# Slice the dataframe
df_trimmed = df.loc[trim_start:trim_end]

# Save the perfectly trimmed active era
df_trimmed.to_csv("v15_master_telemetry_active.csv")

print(f"\n✅ TRIMMING COMPLETE.")
print(f"Original size: {len(df)} epochs")
print(f"Trimmed size: {len(df_trimmed)} epochs")
print("The dataset is now mathematically dense and ready for PyTorch.")