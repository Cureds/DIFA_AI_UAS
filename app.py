import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import librosa
import numpy as np
from PIL import Image
import os

# --- LUXURY DESIGN (DIFA) ---
st.set_page_config(page_title="DIFA", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #0f1116; color: #e0c097; }
    h1 { color: #d4af37; font-family: 'Playfair Display', serif; font-weight: 700; text-align: center; }
    .stButton>button { background-color: #d4af37; color: black; border-radius: 20px; border: none; }
    .stFileUploader { border: 1px dashed #d4af37; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("DIFA")
st.subheader("AI Music Detection")
st.write("---")

# --- 1. LOAD THE BRAIN ---
@st.cache_resource
def load_difa_brain():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Recreate the body (ResNet18)
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    # Load the brain
    model.load_state_dict(torch.load("sonic_guard_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

def process_audio(audio_file):
    # Load whole file and normalize
    audio, _ = librosa.load(audio_file, sr=16000)
    audio = librosa.util.normalize(audio) # ADD THIS LINE
    
    mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Scale to [0, 1] - MUST MATCH TRAINING
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    
    # Prepare for ResNet
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    img = transform(mel_db)
    img = img.repeat(3, 1, 1).unsqueeze(0) 
    return img

# --- 3. THE INTERFACE ---
uploaded_file = st.file_uploader("Upload audio file (mp3, wav, etc.)", type=["mp3", "wav", "m4a", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    with st.spinner("DIFA is analyzing frequencies..."):
        # Predict
        model, device = load_difa_brain()
        input_tensor = process_audio(uploaded_file).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            confidence, prediction = torch.max(probabilities, dim=0)
            
        # Display Result
        label = "AI GENERATED" if prediction.item() == 1 else "AUTHENTIC REAL"
        color = "#ff4b4b" if label == "AI GENERATED" else "#00ff7f"
        
        st.markdown(f"### Result: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
        st.write(f"**Confidence Level:** {confidence.item()*100:.2f}%")
        st.progress(confidence.item())

st.markdown("---")
st.caption("DIFA © 2026 | Applied AI")