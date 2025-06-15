import os                      
import numpy as np             
import librosa                 
import tensorflow as tf       
from sklearn.preprocessing import LabelEncoder    
from sklearn.model_selection import train_test_split  
from sklearn.utils.class_weight import compute_class_weight  
from sklearn.metrics import classification_report, confusion_matrix  
import pickle                  
import warnings
warnings.filterwarnings("ignore")   


DATA_DIR  = "data"     
SR = 16000   # target sample-rate
N_MELS = 64         
MAX_FRAMES = 150        
BATCH_SIZE = 32 
EPOCHS = 100    
LEARNING_RATE = 0.001  # initial learning rate
SEED = 12


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(f"GPU initialisation error: {e}")

# emotions mapping 
EMOTION_MAPPING = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry",  "06": "fearful", "07": "disgust", "08": "surprise",
}

# extracting the features 
def extract_features(audio_path: str, sr: int = SR):
    try:
        y, original_sr = librosa.load(audio_path, sr=None)
        if original_sr != sr:
            y = librosa.resample(y, orig_sr=original_sr, target_sr=sr)

        # removing trailing silence
        y, _ = librosa.effects.trim(y, top_db=30)  
        if len(y) == 0:
            return None

        S = librosa.feature.melspectrogram( y=y, sr=sr, n_mels=N_MELS, hop_length=512, n_fft=1024)

        # scaling
        S = librosa.power_to_db(S, ref=np.max)     
        S = (S - S.mean()) / (S.std() + 1e-6)       

        if S.shape[1] < MAX_FRAMES:
            pad = MAX_FRAMES - S.shape[1]
            S = np.pad(S, ((0, 0), (0, pad)), mode="constant", constant_values=S.min())
        else:
            S = S[:, :MAX_FRAMES]

        return S
    except Exception as e:
        print(f" Feature-extraction error for {audio_path}: {e}")
        return None

# loading the RAVDESS dataset
def load_data(data_dir: str):
    
    print(" Loading the data . . .")
    feats, labels = [], []
    file_count = 0

    for actor in sorted(os.listdir(data_dir)):
        actor_path = os.path.join(data_dir, actor)
        if not os.path.isdir(actor_path):
            continue

        for fname in sorted(os.listdir(actor_path)):
            if not fname.endswith(".wav") or fname.startswith("."):
                continue
            parts = fname.split("-")
            if len(parts) != 7:
                continue

            emo = EMOTION_MAPPING.get(parts[2])
            if emo is None:
                continue

            feat = extract_features(os.path.join(actor_path, fname))
            if feat is not None:
                feats.append(feat)
                labels.append(emo)
                file_count += 1

    if not feats:
        raise RuntimeError("No audio files found")

    X = np.array(feats)
    y = np.array(labels)
    print(f" Loaded {file_count} files")
    return X, y


# defining the model : 2d CNN for speech-emotion classification
def create_improved_model():
    inp = tf.keras.layers.Input(shape=(N_MELS, MAX_FRAMES, 1))

    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    out = tf.keras.layers.Dense(8, activation="softmax")(x)
    return tf.keras.Model(inp, out)


# simple data augmentation ( duplicating the factor times and adding random Gaussian noise )
def augment_data(X, y, factor: int = 2):
    X_aug, y_aug = [], []
    for i in range(len(X)):
        X_aug.append(X[i]); y_aug.append(y[i])
        for _ in range(factor - 1):
            noise = np.random.normal(0, 0.005, X[i].shape)
            X_aug.append(X[i] + noise); y_aug.append(y[i])
    return np.array(X_aug), np.array(y_aug)


np.random.seed(SEED)
tf.random.set_seed(SEED)

audio_X, y_str = load_data(DATA_DIR)
X = audio_X[..., np.newaxis]                       

le = LabelEncoder()
y_int = le.fit_transform(y_str)
y_onehot = tf.keras.utils.to_categorical(y_int, num_classes=8)

X_train, X_test, y_train, y_test = train_test_split( X, y_onehot, test_size=0.2, stratify=y_int, random_state=SEED)
X_train, y_train = augment_data(X_train, y_train, factor=3)

# weight calculation
y_train_int = np.argmax(y_train, axis=1)
cw = compute_class_weight(class_weight="balanced", classes=np.unique(y_train_int), y=y_train_int)
class_weight_dict = dict(enumerate(cw))

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# compling the model 
model = create_improved_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),loss="categorical_crossentropy",
    metrics=["accuracy"],)

#training the model 
history = model.fit( X_train, y_train, validation_split=0.20, epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=8
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "best_model.keras", save_best_only=True
        ),
    ],
    class_weight=class_weight_dict,
)

# final evualtion 
best = tf.keras.models.load_model("best_model.keras")
loss, acc = best.evaluate(X_test, y_test)
print(f"Accuracy: {acc * 100:.2f}%")

best.save("fast_emotion_model.keras")
