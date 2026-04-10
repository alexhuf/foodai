# FoodAI V4: Deep Generative Plate Oracle 🧬
**Master Developer Documentation & Architectural Blueprint**

## 1. Project Overview
FoodAI V4 is a generative machine learning architecture designed to mathematically predict and construct complete plates of food based on an individual's real-time metabolic, psychological, and environmental state. 

Moving beyond traditional categorical classification, this system utilizes a **Transformer-style Attention Mechanism** (Deep Multi-Label Neural Network) to independently calculate the biological probability of over 300+ specific ingredients co-occurring on a plate. The architecture is explicitly optimized to fully utilize an NVIDIA RTX 5070 Ti via native CUDA acceleration, executing multi-hour hyperparameter optimization crucibles.

---

## 2. Hardware & Environment Specifications
* **OS:** Windows 10
* **GPU:** NVIDIA RTX 5070 Ti
* **Native Drivers:** NVIDIA CUDA Toolkit 13.2
* **Deep Learning Framework:** PyTorch (Nightly Build - `cu126` or source-compiled for 50-series architecture)
* **Optimization Framework:** Optuna
* **Explainability Framework:** SHAP (SHapley Additive exPlanations) via XGBoost

---

## 3. Data Architecture & Feature Space
The pipeline ingests raw, year-long historical data (`android_food_entries.csv`, `actions.csv`, `daily_calorie_budgets.csv`) and compiles a massive **100+ Dimensional Matrix** to eliminate all behavioral blind spots.

### 3.1 Chronobiology & Temporal Sequencing
* `Fasting_Window_Minutes`: Exact duration since the last caloric intake.
* `Meal_Time_Slot`: Standardized daily intake windows.
* `Cyclical_Time`: Day of Week, Week of Month, and Month of Year (encoded via sine/cosine transformations to preserve continuous temporal relationships).
* `Weekend_Holiday_Proximity`: Boolean flags for variations in standard routine.

### 3.2 Deep Metabolic State
* `Intraday_Caloric_Velocity`: Cumulative calories consumed prior to the current meal.
* `Remaining_Intraday_Budget`: Absolute calorie allowance remaining for the calendar day.
* `Macro_Starvation_Vectors`: 3-day, 7-day, 14-day, and 30-day rolling averages for Carbohydrates, Fats, and Proteins.
* `Willpower_Fatigue`: Continuous count of consecutive days adhering to a caloric deficit.

### 3.3 Physical Fatigue & Body Composition
* `Weight_Velocity_7D` / `Weight_Velocity_30D`: Rolling delta of body mass indicating hormonal shifts (e.g., ghrelin/leptin response).
* `Acute_Fatigue`: 3-day rolling step count average.
* `Chronic_Fatigue`: 30-day rolling step count average.

### 3.4 Environmental Stress
* `Daylight_Hours`: Tracks seasonal affective behavioral shifts.
* `Approx_High_Temp`: Daily thermal highs.
* `Precipitation_Flag`: Boolean tracking rain/snow events (highly correlated with cooking vs. delivery).

### 3.5 The Target Matrix (Y)
The dataset is collapsed from a "Line-Item Ledger" into "Full Plates." The target variable is not a single string, but a **Multi-Label One-Hot Encoded Vector**. Every unique ingredient in the user's history occupies its own independent column (e.g., `has_chicken`, `has_rice`, `has_sourdough`).

---

## 4. The Execution Pipeline
The repository is structured into three distinct execution scripts. They must be run sequentially.

### Script 1: `1_matrix_compiler.py` (ETL & Engineering)
**Purpose:** Transforms raw historical logs into the 100-dimensional training matrix.
* Merges `entries`, `actions`, and `budgets` based on precise Unix timestamps.
* Calculates all rolling averages and temporal deltas.
* Collapses itemized foods into plate-level arrays.
* **Output:** `v4_master_training_matrix.csv`

### Script 2: `2_shap_audit.py` (Mathematical Explainability)
**Purpose:** A diagnostic run to prove relationship causality before deep training.
* Trains a rapid, highly aggressive XGBoost regressor on the V4 Matrix.
* Applies SHAP game-theory mathematics to calculate the exact percentage of impact each of the 100 features has on specific cravings.
* **Output:** Generates a visual relationship audit (Dependency plots, Summary plots) to ensure no data leakage or feature blind spots exist.

### Script 3: `3_deep_oracle_training.py` (The 5070 Ti Crucible)
**Purpose:** The core generative training loop.
* **Architecture:** A PyTorch Neural Network featuring self-attention layers to map the co-occurrence relationships of ingredients (e.g., predicting that "Sourdough" inherently increases the probability of "Deli Ham").
* **Loss Function:** `BCEWithLogitsLoss` (Binary Cross Entropy) optimized for independent multi-label generation.
* **Compute Strategy:** Wrapped in an `optuna.create_study()`. The script will spawn hundreds of network variations (layer depths, learning rates, dropouts) and train them simultaneously on the GPU over tens of thousands of epochs.
* **Output:** `v4_deep_generator.pth` (The frozen neural weights) and `v4_env_objects.pkl` (Encoders, Scalers, and Food Vocabulary).

---

## 5. Inference & The User Interface
**Script:** `app.py` (Streamlit Dashboard)
* **Background Process:** Upon launch, the app ingests the user's absolute latest historical log to automatically populate the rolling 30-day biological baseline.
* **User Input:** The UI exposes only highly volatile, immediate inputs (Current Fasting Hours, Current Temperature, Remaining Budget).
* **Generation:** The PyTorch model executes a forward pass through the attention layers. Any output neuron exceeding a calibrated probability threshold (e.g., > 0.5) is activated. The UI aggregates the activated neurons and prints the mathematically hallucinated plate.

---

## 6. Auditor Notes
* **Data Leakage:** Pay close attention to `1_matrix_compiler.py`. Ensure that rolling averages strictly utilize data from `T-1` backwards. The model must not see future data when predicting a meal at time `T`.
* **Hardware Utilization:** Monitor `nvidia-smi` during the execution of Script 3. The `CUDA_LAUNCH_BLOCKING` environment variable may be adjusted depending on the stability of the PyTorch Nightly build on the 50-series architecture.
* **Vocabulary Sparsity:** If the user has logged 1,000 unique foods but only eaten 800 of them once, the output vector will be highly sparse. Consider implementing a minimum frequency threshold (e.g., `min_occurrences=3`) during the mapping phase in Script 1 to prevent the network from memorizing outliers.