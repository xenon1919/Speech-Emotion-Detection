
# Speech Emotion Detection

A web application that identifies emotions from speech audio files. This project uses machine learning to predict emotions based on audio features extracted from `.wav` files.

## Features

- Supports `.wav` audio files.
- Predicts emotions such as **Neutral**, **Calm**, **Happy**, **Sad**, **Angry**, **Fearful**, **Disgust**, and **Surprised**.
- Interactive user interface built with Streamlit.

## Project Structure

```
Speech-Emotion-Detection/
├── app.py                      # Streamlit app for emotion detection
├── speech-emotion-detection.ipynb  # Jupyter notebook for model training and experimentation
├── trained_model.pkl           # Pretrained machine learning model
├── README.md                   # Project description and setup instructions
├── requirements.txt            # Dependencies for the project
├── temp/                       # Temporary directory for uploaded audio files
└── .gitignore                  # To ignore unnecessary files
```

## Getting Started

### Prerequisites

- Python 3.8 or above
- Recommended: Virtual environment for dependency management

### Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-URL>
   cd Speech-Emotion-Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the `trained_model.pkl` file is in the repository root.

### Running the Application

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open the local URL in your browser (e.g., `http://localhost:8501`).

3. Upload a `.wav` file to predict its emotion.

### Training the Model

The `speech-emotion-detection.ipynb` file contains code for training and testing the emotion detection model. You can use this notebook to experiment with the dataset or retrain the model.


## Built With

- [Streamlit](https://streamlit.io/)
- [Librosa](https://librosa.org/)
- [NumPy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/)


## Acknowledgments

- Emotion dataset and pretrained model sources
- Inspiration and tutorials for audio feature extraction

---
