import pickle
import pandas as pd
import numpy as np
import streamlit as st
import gdown
import os
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# --- Page configuration ---
st.set_page_config(page_title="Stress Prediction", layout="centered")

# --- Logo ---
st.image("Logo.jpg", width=100)

# --- Custom CSS for Background and Fonts ---
st.markdown("""
    <style>
        body {
            background-color: #111;
            color: #f0f0f0;
        }
        .stTextInput > div > div > input {
            background-color: #222;
            color: white;
        }
        .stNumberInput > div > div > input {
            background-color: #222;
            color: white;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        h1 {
            text-align: center;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🤖 Stress Prediction Using ML")

# --- Download and load data & model training (only once) ---
FILE_ID = "1xQT5B7gjBUWagFw96KOOJV4B8X5dIPCn"
FILE_NAME = "S4.pkl"

if not os.path.exists(FILE_NAME):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, FILE_NAME, quiet=False)

with open(FILE_NAME, 'rb') as file:
    data = pickle.load(file, encoding='latin1')

signal_data = data['signal']['chest']
labels = data['label']
eda = signal_data['EDA'].squeeze()
temp = signal_data['Temp'].squeeze()
resp = signal_data['Resp'].squeeze()

min_len = min(len(eda), len(temp), len(resp), len(labels))
eda = eda[:min_len]
temp = temp[:min_len]
resp = resp[:min_len]
labels = labels[:min_len]

combined = np.stack([eda, resp, temp], axis=1)
combined_downsampled = combined[::50]
labels_downsampled = labels[::50]

mask = np.isin(labels_downsampled, [1, 2, 3])
filtered_data = combined_downsampled[mask]
filtered_labels = labels_downsampled[mask]
filtered_data = np.squeeze(filtered_data)

df = pd.DataFrame(filtered_data, columns=['EDA', 'Resp', 'Temp'])
df['label'] = filtered_labels

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

X = df[['EDA', 'Resp', 'Temp']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# --- Input fields ---
st.subheader("Enter physiological values:")
eda_input = st.number_input("EDA", min_value=0.0, value=0.30, step=0.01)
resp_input = st.number_input("Respiration", min_value=0.0, value=5.50, step=0.1)
temp_input = st.number_input("Temperature", min_value=0.0, value=40.60, step=0.1)

# --- Prediction and Download ---
if st.button("Predict Stress Level"):
    input_data = np.array([[eda_input, resp_input, temp_input]])
    pred = model.predict(input_data)[0]
    
    reverse_le = dict(zip(le.transform(le.classes_), le.classes_))
    original_label = reverse_le[pred]
    
    label_map = {
        1: "No Stress (Baseline)",
        2: "Stress",
        3: "No Stress (Amusement)"
    }
    result = label_map.get(original_label, "Unknown")
    
    if result == "Stress":
        st.error(f"Predicted Result: {result}")
    else:
        st.success(f"Predicted Result: {result}")
    
    # Prepare content for download
    filename = f"stress_prediction_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    content = f"""STRESS PREDICTION RESULT

Temperature: {temp_input}
EDA: {eda_input}
Respiration: {resp_input}

Predicted Result: {result}
"""
    st.download_button(
        label="Download Result as Text File",
        data=content,
        file_name=filename,
        mime="text/plain"
    )
