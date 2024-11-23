import streamlit as st
import librosa
import numpy as np
import os
import pickle

# Define emotion mapping as per your model's training
emotions = {
    1: "Neutral",
    2: "Calm",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Fearful",
    7: "Disgust",
    8: "Surprised"
}

# Load your trained model
MODEL_PATH = "trained_model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
else:
    model = None
    st.error("Model file not found!")

# Function to extract MFCC features from an audio file
def extract_mfcc(file_path):
    try:
        y, sr = librosa.load(file_path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_features = np.mean(mfcc.T, axis=0)
        return mfcc_features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# Streamlit app layout
st.title("Speech Emotion Recognition")
st.write("Upload a .wav audio file to predict its emotion.")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract MFCC features
    st.info("Extracting features...")
    features = extract_mfcc(temp_path)

    if features is not None:
        # Reshape features for prediction (matching the input shape during training)
        features = np.reshape(features, newshape=(1, 40, 1))

        # Make prediction
        if model:
            predictions = model.predict(features)
            # Get the emotion with the highest prediction probability
            predicted_emotion = emotions[np.argmax(predictions[0]) + 1]
            st.success(f"Prediction: {predicted_emotion}")
        else:
            st.error("Model not loaded!")

    # Clean up temporary file
    os.remove(temp_path)
