# Sstreamlit interface 

import io, tempfile, numpy as np, pandas as pd, librosa, soundfile as sf
import streamlit as st, tensorflow as tf, altair as alt
from packaging import version
import pickle


st.set_page_config( page_title="Speech Emotion Classifier",layout="centered",page_icon="🎙️",)


SR = 16_000
N_MELS = 64
MAX_FRAMES = 150
MODEL_PATH = "fast_emotion_model.keras"
ENCODER_PATH  = "label_encoder.pkl"
LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprise",
]

# loading the model 
@st.cache_resource
def load_model_and_encoder():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)

        # loading
        try:
            with open(ENCODER_PATH, "rb") as f:
                encoder = pickle.load(f)
            labels = encoder.classes_.tolist()
        except FileNotFoundError:
            # labeling
            labels = LABELS
            encoder = None

        return model, labels, encoder

    except Exception as e:
        st.error(f"Error while loading the model: {e}")
        return None, LABELS, None


net, emotion_labels, label_encoder = load_model_and_encoder()


# preprocessing
def preprocess_audio(y: np.ndarray, sr: int) -> np.ndarray:
    try:
        # checking the input
        if len(y) == 0:
            raise ValueError("Empty audio signal")

        if sr != SR:
            if version.parse(librosa.__version__) >= version.parse("0.10"):
                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            else:
                y = librosa.resample(y, sr, SR)
            sr = SR

        # trailing the silence
        y, _ = librosa.effects.trim(y, top_db=30)
        if len(y) == 0:
            raise ValueError("Audio too short after silence removal")

        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, hop_length=512, n_fft=1024
        )
        S = librosa.power_to_db(S, ref=np.max)

        # normalisation
        S = (S - S.mean()) / (S.std() + 1e-6)

        if S.shape[1] < MAX_FRAMES:
            pad_width = MAX_FRAMES - S.shape[1]
            S = np.pad(S, ((0, 0), (0, pad_width)), mode="constant",constant_values=S.min(),)
        else:
            S = S[:, :MAX_FRAMES]

        return S[np.newaxis, ..., np.newaxis].astype("float32")

    except Exception as e:
        st.error(f"Pre-processing error: {e}")
        return None


#loading the audion file 
def load_audio_file(file_data):
    try:
        y, sr = sf.read(io.BytesIO(file_data), always_2d=False)

        # Converting stereo to mono
        if y.ndim > 1:
            y = y.mean(axis=1)

        return y, sr

    except Exception:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
                tmp.write(file_data)
                tmp.flush()
                y, sr = librosa.load(tmp.name, sr=None)
                return y, sr
        except Exception as e:
            st.error(f"Unable to load audio file: {e}")
            return None, None


#user interface 
st.title("🎙️ Speech Emotion Recognition")
st.markdown("---")

# Model information
with st.sidebar:
    st.header("ℹ️ Informations: ")
    st.write("**Supported formats:** WAV, MP3, OGG, FLAC")

    st.write("**Detected emotions:**")
    for i, emotion in enumerate(emotion_labels, 1):
        st.write(f"{i}. {emotion.capitalize()}")

    st.markdown("---")
    st.write("**Model parameters:**")
    st.write(f"- Sample rate: {SR} Hz")
    st.write(f"- Mel filters: {N_MELS}")
    st.write(f"- Max frames: {MAX_FRAMES}")


st.subheader("📁 Upload an audio file")
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "ogg", "flac"],
    help="Upload an audio file to analyse its emotion",
)

if uploaded_file is not None:
    st.subheader("🔊 Audio preview")
    st.audio(uploaded_file, format="audio/wav")

    # File details
    file_details = {
        "Name": uploaded_file.name,
        "Size": f"{len(uploaded_file.getvalue()) / 1024:.1f} KB",
        "Type": uploaded_file.type,
    }

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File name", file_details["Name"])
    with col2:
        st.metric("Size", file_details["Size"])
    with col3:
        st.metric("Format", file_details["Type"])

    # prediction button
    if st.button(" Analyse emotion", type="primary"):
        if net is None:
            st.error("Model not available !! ")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                #file loading 
                status_text.text("📥 Loading audio file…")
                progress_bar.progress(25)

                file_data = uploaded_file.getvalue()
                y, sr = load_audio_file(file_data)

                if y is None:
                    st.error("❌ Could not load the audio file")
                else:
                    # precprocessing 
                    status_text.text("🔄 Pre-processing audio…")
                    progress_bar.progress(50)

                    X = preprocess_audio(y, sr)

                    if X is None:
                        st.error("❌ Pre-processing failed")
                    else:
                        # prediction
                        status_text.text("🧠 Running inference…")
                        progress_bar.progress(75)

                        probabilities = net.predict(X, verbose=0)[0]
                        predicted_idx = int(np.argmax(probabilities))
                        predicted_emotion = emotion_labels[predicted_idx]
                        confidence = probabilities[predicted_idx] * 100

                        progress_bar.progress(100)
                        status_text.text("✅ Analysis complete!")

                        # final results
                        st.markdown("---")
                        st.subheader("📊 Analysis results")

                        st.success(f"**🎯 Detected emotion: {predicted_emotion.upper()}**")
                        st.info(f"**Confidence: {confidence:.1f}%**")

                        # Probability chart
                        st.subheader("📈 Probability distribution")

                        df_probs = pd.DataFrame(
                            {
                                "Emotion": [e.capitalize() for e in emotion_labels],
                                "Probability": probabilities,
                                "Color": [
                                    "#ff6b6b" if i == predicted_idx else "#4ecdc4"
                                    for i in range(len(emotion_labels))
                                ],
                            }
                        )

                        chart = (
                            alt.Chart(df_probs)
                            .mark_bar()
                            .encode(
                                x=alt.X("Emotion", sort="-y", title="Emotions"),
                                y=alt.Y(
                                    "Probability",
                                    title="Probability",
                                    scale=alt.Scale(domain=[0, 1]),
                                ),
                                color=alt.Color("Color", scale=None),
                                tooltip=[
                                    "Emotion",
                                    alt.Tooltip("Probability", format=".3f"),
                                ],
                            )
                            .properties(
                                height=400,
                                title="Probability distribution per emotion",
                            )
                        )

                        st.altair_chart(chart, use_container_width=True)

                        # Detailed table
                        with st.expander("📋 Probability details"):
                            df_detailed = pd.DataFrame(
                                {
                                    "Emotion": [e.capitalize() for e in emotion_labels],
                                    "Probability": [f"{p:.3f}" for p in probabilities],
                                    "Percentage": [f"{p*100:.1f}%" for p in probabilities],
                                }
                            )
                            st.dataframe(df_detailed, use_container_width=True)

                        progress_bar.empty()
                        status_text.empty()

            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")
                progress_bar.empty()
                status_text.empty()

else:
    # landing page message
    st.info( "⬆️ Upload a WAV, MP3, OGG, or FLAC file to start the emotion-analysis process.")

    # usage tips
    with st.expander("💡 Usage tips"):
        st.write(
            """
            **For best results:**
            - Use clear recordings with minimal background noise.
            - Aim for clips that are a few seconds long (2 to 10 s).
            - Make sure speech is clearly audible.
            - WAV generally offers the highest quality.

            **Supported emotions:**
            The model distinguishes eight different emotions.
            """
        )

# footer
st.markdown("---")
st.markdown("*Speech-emotion recognition app*")
