Project goal : Recognising emotions from short speech clips (8 emotions: "neutral", "calm", "happy", "sad",
"angry", "fearful", "disgust", "surprise").

Dataset : RAVDESS 1440 acted utterances, 8 emotions.

Audio processing
1. Sampling every clip to 16 kHz
2. Cutting the silence at the start and end.
3. Turning the sound wave into a Mel-spectrogram .
4. Normalzing the spectrogram.
5. Pading or croping the time axis to 150 frames.
6. creating two extra copies of each training sample with low‑level Gaussian noise.
7. Applying class‑balanced weights during training.

Model used : 
 2d CNN network
  • 3 convolutional blocks  
  • Global‑average pooling  
  • 2 dense layers  
  • 8 class soft‑max output  
 Early stopping for preventing over‑fitting.

Files delivered: 
- train.py – model training source code 
- fast_emotion_model.keras – model for inference  
- label_encoder.pkl – maps class idx ⇒ emotion label  
- app.py – Streamlit demo UI


Requirements: 
tensorflow>=2.10  
librosa>=0.10  
numpy  
scikit‑learn  
streamlit  
streamlit‑webrtc  
