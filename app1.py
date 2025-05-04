import streamlit as st
import librosa
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import tempfile
from audio_recorder_streamlit import audio_recorder

# Set page config
st.set_page_config(
    page_title="Audio Classification App",
    page_icon="🎧",
    layout="wide"
)

# Load model and label encoder
@st.cache_resource
def load_ml_components():
    model = load_model('saved_models/best_audio_classification_final.keras')
    labelencoder = LabelEncoder()
    classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
               'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
               'siren', 'street_music']  # Update with your actual classes
    labelencoder.fit(classes)
    return model, labelencoder

model, labelencoder = load_ml_components()

# Feature extraction function
def features_extractor(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sample_rate)
        features = np.vstack([mfccs, chroma, mel, contrast, tonnetz])
        return np.mean(features.T, axis=0)
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# Prediction function
def predict_audio_class(file_path):
    features = features_extractor(file_path)
    if features is not None:
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        predicted_label = np.argmax(prediction, axis=1)
        return labelencoder.inverse_transform(predicted_label)[0]
    return "Error in processing"

# Main app
st.title("AudibleEyes")

# Create tabs
tab1, tab2 = st.tabs(["📁 Upload Audio", "🎤 Record Audio"])

with tab1:
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])
    if uploaded_file is not None:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        # Play audio
        st.audio(uploaded_file)
        
        # Classify button
        if st.button("Classify Uploaded Audio"):
            with st.spinner("Analyzing..."):
                prediction = predict_audio_class(tmp_path)
                st.success(f"Predicted class: {prediction}")
        
        # Clean up
        os.unlink(tmp_path)

with tab2:
    st.write("Record audio (15 seconds max):")
    
    # Audio recorder widget
    audio_bytes = audio_recorder(
        energy_threshold=(-1.0, 1.0),
        pause_threshold=15.0,
        sample_rate=44100,
        text="Click to record",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="2x",
    )
    
    if audio_bytes:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        # Playback
        st.audio(audio_bytes, format="audio/wav")
        
        # Separate classify button
        if st.button("Classify Recording"):
            with st.spinner("Analyzing recording..."):
                prediction = predict_audio_class(tmp_path)
                st.success(f"Predicted class: {prediction}")
        
        # Clean up
        os.unlink(tmp_path)

# Add some information about the model
st.sidebar.header("ℹ️ About")
st.sidebar.info("""
This app uses a deep learning model to classify audio into different categories.
The model was trained on an audio dataset with the following classes:
- Air Conditioner
- Car Horn
- Children Playing
- Dog Bark
- Drilling
- Engine Idling
- Gun Shot
- Jackhammer
- Siren
- Street Music
""")
st.sidebar.markdown("---")
st.sidebar.markdown("🛠️ **Technical Details**")
st.sidebar.markdown("- Uses MFCC, Chroma, Mel Spectrogram features")
st.sidebar.markdown("- Deep neural network with 3 hidden layers")
st.sidebar.markdown("- Trained with dropout and batch normalization")
