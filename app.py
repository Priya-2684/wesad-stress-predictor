import pickle
import pandas as pd
import numpy as np
import streamlit as st
import gdown
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Download from Google Drive if not exists
FILE_ID = "1xQT5B7gjBUWagFw96KOOJV4B8X5dIPCn"
FILE_NAME = "S4.pkl"

if not os.path.exists(FILE_NAME):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, FILE_NAME, quiet=False)

# Load WESAD Data
with open(FILE_NAME, 'rb') as file:
    data = pickle.load(file, encoding='latin1')

# Extract signal
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

# Streamlit UI
st.title("🤖 Stress Prediction Using WESAD (S11)")

st.write("Enter physiological values:")

eda_input = st.number_input("EDA", min_value=0.0)
resp_input = st.number_input("Respiration", min_value=0.0)
temp_input = st.number_input("Temperature", min_value=0.0)

if st.button("Predict Stress Level"):
    input_data = np.array([[eda_input, resp_input, temp_input]])
    pred = model.predict(input_data)[0]
    
    label_map = {
        1: "No Stress (Baseline)",
        2: "Stress",
        3: "No Stress (Amusement)"
    }
    
    reverse_le = dict(zip(le.transform(le.classes_), le.classes_))
    original_label = reverse_le[pred]
    result = label_map.get(original_label, "Unknown")
    
    st.success(f"Predicted Result: **{result}**")
