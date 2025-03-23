import streamlit as st
import librosa
import numpy as np
import os
import pickle
import tempfile  # For handling uploaded files

# Define emotion mapping (update according to your model's labels)
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

# Load the trained model
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
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_features = np.mean(mfcc.T, axis=0)  # Convert to 1D feature vector
        return mfcc_features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# Streamlit UI
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a `.wav` audio file to predict its emotion.")

uploaded_file = st.file_uploader("Upload an Audio File", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.getbuffer())
        temp_path = temp_audio.name  # Get the path of the temp file

    # Extract features
    st.info("Extracting features...")
    features = extract_mfcc(temp_path)

    if features is not None:
        # Reshape features to match model input shape
        features = np.reshape(features, (1, -1))  # (1, 40) shape

        if model:
            # Predict emotion
            predictions = model.predict(features)
            predicted_label = np.argmax(predictions) + 1  # Get class index
            predicted_emotion = emotions.get(predicted_label, "Unknown")
            
            st.success(f"🎭 Prediction: {predicted_emotion}")
        else:
            st.error("Model not loaded!")

    # Remove temporary file
    os.remove(temp_path)
