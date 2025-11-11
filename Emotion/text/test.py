import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load trained model
# -----------------------------
model = load_model("emotion_bilstm.h5")
print("✅ Model loaded successfully.")

# -----------------------------
# Load metrics to get class names
# -----------------------------
with open("metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

class_names = metrics['class_names']

# -----------------------------
# Prepare tokenizer
# -----------------------------
# You need the same tokenizer as training
# Here we rebuild it from training texts (or you can save tokenizer as pickle during training)
# For demonstration, let's assume you have a CSV with training texts
data = pd.read_csv("isear.csv")
texts_train = data['text'].astype(str).tolist()

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts_train)

maxlen = 100  # same as training

# -----------------------------
# Function to predict emotion
# -----------------------------
def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding='post')
    pred = model.predict(padded ,verbose=0) 
    label_idx = np.argmax(pred)
    emotion = class_names[label_idx]
    confidence = float(np.max(pred))
    return emotion, confidence

# -----------------------------
# Test examples
# -----------------------------
text="i am dancing"

emotion, conf = predict_emotion(text)
print(f"Text: {text}\nPredicted Emotion: {emotion}, Confidence: {conf:.2f}\n")
