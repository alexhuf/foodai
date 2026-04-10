import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import os

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="FoodAI Oracle", page_icon="🧠", layout="centered")
st.title("🧠 FoodAI: The Metabolic Oracle")
st.write("Adjust your current physical and environmental state below to see what your body is demanding.")

# --- 2. LOAD THE BRAIN ---
@st.cache_resource
def load_models():
    try:
        # Assuming your build_brain.py saved the XGBoost model and LabelEncoder as .pkl files
        with open("xgboost_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        return model, le
    except FileNotFoundError:
        st.warning("⚠️ Waiting for brain files. Have you run `python build_brain.py` yet?")
        return None, None

model, le = load_models()

# --- 3. UI DASHBOARD INPUTS ---
st.sidebar.header("Current Conditions")

# Sliders for your specific dataset variables
current_steps = st.sidebar.slider("Steps Taken Today", 0, 30000, 5000)
high_temp = st.sidebar.slider("Outside High Temp (°F)", 0, 100, 50)
calorie_deficit = st.sidebar.slider("Calorie Deficit (Last 3 Days)", -1000, 3000, 0)
daylight_hours = st.sidebar.slider("Daylight Hours", 8.0, 16.0, 12.0)

# Automatic Boolean Flags based on your temp slider
is_freezing = 1 if high_temp <= 32 else 0
is_hot = 1 if high_temp >= 75 else 0

# --- 4. PREDICTION ENGINE ---
if st.button("🔮 Consult the Oracle", use_container_width=True):
    if model is not None and le is not None:
        
        # Package the inputs exactly how XGBoost expects them
        # (Ensure this order matches how you trained your model in build_brain.py)
        input_data = np.array([[
            current_steps, 
            calorie_deficit, 
            high_temp, 
            daylight_hours, 
            is_freezing, 
            is_hot
        ]])
        
        # Get the mathematical prediction
        encoded_prediction = model.predict(input_data)
        prediction = le.inverse_transform(encoded_prediction)[0]
        
        st.divider()
        st.header(f"🧠 AI Target Profile: {prediction}")
        
        # --- 5. THE REVERSE LOOKUP ENGINE ---
        try:
            # Load your historical dataset
            df = pd.read_csv('Ultimate_Super_Dataset.csv')
            
            # The exact column names verified from your dataset
            category_column = 'Meal_Archetype'
            food_column = 'Display_Name'
            
            # Filter the dataset for days where you ate this exact archetype
            matching_days = df[df[category_column] == prediction]
            
            # Extract the unique, actual meal combinations from those days
            historical_meals = matching_days[food_column].dropna().unique().tolist()
            
            st.subheader("🍽️ Actionable Menu Suggestions")
            st.write("Based on your physical state and the weather, here are the exact dishes from your history that satisfy this craving:")
            
            if historical_meals:
                # Randomly select up to 5 real meals to suggest so the menu feels fresh
                suggestions = random.sample(historical_meals, min(5, len(historical_meals)))
                
                for meal in suggestions:
                    st.success(f"**Option:** {meal}") 
            else:
                st.info("No specific food strings found for this category in the database.")

        except FileNotFoundError:
            st.error("Could not locate Ultimate_Super_Dataset.csv for reverse lookup.")
        except Exception as e:
            st.error(f"Lookup error: {e}")

    else:
        st.error("Model not loaded. Please train the AI first.")