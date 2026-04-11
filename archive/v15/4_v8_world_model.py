import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle

print("🌍 V8 WORLD MODEL: FORGING THE BIOLOGICAL SIMULATOR...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Computing Transition Dynamics on {device.type.upper()}")

# --- 1. LOAD THE MATRIX ---
df = pd.read_csv('v5_omni_matrix.csv')

# Isolate the 157 numerical features
exclude_cols = ['date', 'timeSlot', 'Meal_Time', 'Display_Name', 'Exact_Time', 'Sequence_Target_String']
feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.int64, np.int32, float, int]]

# Fill any stray NaNs
df[feature_cols] = df[feature_cols].fillna(0)

# --- 2. CREATE T -> T+1 TRANSITION PAIRS ---
print("⏱️ Mapping Temporal Cause-and-Effect Sequences...")
# We need to predict tomorrow's state based on today's state
df['date_obj'] = pd.to_datetime(df['date'])
daily_states = df.groupby('date_obj')[feature_cols].mean().reset_index()

X_temporal = []
y_temporal = []

for i in range(len(daily_states) - 1):
    current_day = daily_states.iloc[i]
    next_day = daily_states.iloc[i+1]
    
    # Ensure they are actually consecutive days (don't map Monday to Friday if data is missing)
    if (next_day['date_obj'] - current_day['date_obj']).days == 1:
        X_temporal.append(current_day[feature_cols].values)
        y_temporal.append(next_day[feature_cols].values)

X_temporal = np.array(X_temporal, dtype=np.float32)
y_temporal = np.array(y_temporal, dtype=np.float32)

print(f"Total Valid Day-to-Day Transitions Mapped: {len(X_temporal)}")

# Standardize the physics
mean_state = np.mean(X_temporal, axis=0)
std_state = np.std(X_temporal, axis=0) + 1e-8 # Prevent division by zero

X_scaled = (X_temporal - mean_state) / std_state
y_scaled = (y_temporal - mean_state) / std_state

class TransitionDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

dataloader = DataLoader(TransitionDataset(X_scaled, y_scaled), batch_size=16, shuffle=True)

# --- 3. THE BIOLOGICAL PHYSICS ENGINE ---
class TemporalWorldModel(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        # A deep, dense network to learn the complex non-linear physics of metabolism
        self.physics_engine = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.Mish(), # Mish activation is excellent for continuous physics simulations
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, state_dim) # Predicts the exact same 157 dimensions for tomorrow
        )

    def forward(self, state_t):
        # The model predicts the DELTA (change), not the absolute value. 
        # This makes it mathematically stable over 5-day simulations.
        state_delta = self.physics_engine(state_t)
        return state_t + state_delta

model = TemporalWorldModel(state_dim=len(feature_cols)).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

# --- 4. TRAINING THE MATRIX ---
print("⚔️ Igniting the Simulator Crucible (Learning your biology)...")
epochs = 300
model.train()

for epoch in range(epochs):
    total_loss = 0
    for state_t, state_next in dataloader:
        state_t, state_next = state_t.to(device), state_next.to(device)
        
        optimizer.zero_grad()
        predicted_next = model(state_t)
        
        loss = criterion(predicted_next, state_next)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch+1:03d}/{epochs} | Physics Error (MSE): {total_loss/len(dataloader):.4f}")

# --- 5. FREEZE THE WORLD MODEL ---
print("\n📦 Freezing the Universe...")
torch.save(model.state_dict(), 'v8_world_model_weights.pth')

with open('v8_physics_env.pkl', 'wb') as f:
    pickle.dump({
        'feature_cols': feature_cols,
        'mean_state': mean_state,
        'std_state': std_state
    }, f)

print("✅ THE WORLD MODEL IS ONLINE. The physics of your body have been mapped.")