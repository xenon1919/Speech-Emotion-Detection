
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



## Built With

- [Streamlit](https://streamlit.io/)
- [Librosa](https://librosa.org/)
- [NumPy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/)


## Acknowledgments

- Emotion dataset and pretrained model sources
- Inspiration and tutorials for audio feature extraction

---
