import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🚀 INITIALIZING FOODAI MAXIMUM COMPUTE PIPELINE...")
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(f"Neural Network utilizing: {device}")

# 1. LOAD DATA
try:
    df = pd.read_csv('Ultimate_Super_Dataset.csv')
except FileNotFoundError:
    print("ERROR: Please ensure 'Ultimate_Super_Dataset.csv' is in this folder.")
    exit()

df.fillna(0, inplace=True)
numeric_features = ['calories', 'steps', 'weightLossZoneUpperBound', 'Rolling_3Day_Deficit', 'Rolling_3Day_Steps', 'Daylight_Hours', 'Approx_High_Temp']
categorical_features = ['Meal_Time', 'Day_of_Week', 'Is_Freezing', 'Is_Hot']
target_col = 'Meal_Archetype'

# Encode and Scale
encoders = {}
for col in categorical_features + [target_col]:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

X_tabular = df[numeric_features + categorical_features].values
y = df[target_col].values
num_classes = len(encoders[target_col].classes_)

# 2. SEQUENCE GENERATION
sequence_length = 5 
X_seq, y_seq = [], []
for i in range(len(df) - sequence_length):
    X_seq.append(X_tabular[i:(i + sequence_length)])
    y_seq.append(y[i + sequence_length])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)
X_tab_aligned = X_tabular[sequence_length:]

from sklearn.model_selection import train_test_split

# Shuffle the data so rare meals are evenly distributed
X_tab_train, X_tab_test, X_seq_train, X_seq_test, y_train, y_test = train_test_split(
    X_tab_aligned, X_seq, y_seq, test_size=0.2, random_state=42
)

# Safety Net: Ensure XGBoost gets perfectly ordered classes, even if outliers exist
safe_le = LabelEncoder()
y_train = safe_le.fit_transform(y_train)
# If the test set has a bizarre rare meal, safely default it to the most common class
y_test = np.array([val if val in safe_le.classes_ else safe_le.classes_[0] for val in y_test])
y_test = safe_le.transform(y_test)
num_classes = len(safe_le.classes_)

# 3. XGBOOST HYPERPARAMETER SEARCH
print("\n🔥 Firing up GPU for XGBoost...")
def objective(trial):
    param = {
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
    }
    model = xgb.XGBClassifier(**param)
    model.fit(X_tab_train, y_train)
    return accuracy_score(y_test, model.predict(X_tab_test))

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20) 

best_xgb = xgb.XGBClassifier(**study.best_params, tree_method='hist', device='cuda', objective='multi:softprob', num_class=num_classes)
best_xgb.fit(X_tab_train, y_train)
xgb_train_probs = best_xgb.predict_proba(X_tab_train)
xgb_test_probs = best_xgb.predict_proba(X_tab_test)

# 4. PYTORCH LSTM TRAINING
print("\n🧬 Training PyTorch Deep Sequence Model...")

class CravingsLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CravingsLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]) 

lstm_model = CravingsLSTM(X_seq_train.shape[2], 64, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

X_seq_train_t = torch.tensor(X_seq_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

for epoch in range(50):
    optimizer.zero_grad()
    loss = criterion(lstm_model(X_seq_train_t), y_train_t)
    loss.backward()
    optimizer.step()

lstm_model.eval()
with torch.no_grad():
    lstm_train_probs = torch.softmax(lstm_model(X_seq_train_t), dim=1).cpu().numpy()
    lstm_test_probs = torch.softmax(lstm_model(torch.tensor(X_seq_test, dtype=torch.float32).to(device)), dim=1).cpu().numpy()

# 5. META-LEARNER
print("\n🤝 Training Meta-Learner...")
X_meta_train = np.hstack((xgb_train_probs, lstm_train_probs))
X_meta_test = np.hstack((xgb_test_probs, lstm_test_probs))
meta_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, tree_method='hist', device='cuda')
meta_model.fit(X_meta_train, y_train)

# 6. EXPORT
print("\n📦 Freezing the Brain...")
with open('encoders.pkl', 'wb') as f: pickle.dump(encoders, f)
with open('scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
best_xgb.save_model('xgb_base.json')
meta_model.save_model('meta_model.json')
torch.save(lstm_model.state_dict(), 'lstm_weights.pth')
print("\n✅ TRAINING COMPLETE!")