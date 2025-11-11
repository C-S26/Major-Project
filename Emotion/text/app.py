import os
import cv2
import numpy as np
import pickle
import re
import html
from collections import Counter
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# -------------------------------
# SETTINGS
# -------------------------------
# BiLSTM
BILSTM_MODEL_PATH = "./emotion_bilstm.h5"
TOKENIZER_PATH   = "./tokenizer.pkl"
METRICS_PATH     = "./metrics.pkl"
MAX_LEN          = 100
VOCAB_SIZE       = 10000

# CNN Emotion Detection
CNN_FOLDER       = "./cnn"
CNN_MODEL_PATH   = os.path.join(CNN_FOLDER, "emotion_model.h5")
HAAR_CASCADE     = os.path.join(CNN_FOLDER, "haarcascade_frontalface_default.xml")

# -------------------------------
# LOAD BILSTM MODEL
# -------------------------------
print("🔹 Loading BiLSTM model...")
bilstm_model = load_model(BILSTM_MODEL_PATH)

if os.path.exists(TOKENIZER_PATH):
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    print(f"✅ Loaded tokenizer from {TOKENIZER_PATH}")
else:
    print(f"⚠️ Tokenizer not found, fitting new one")
    data = pd.read_csv("isear.csv")
    texts_train = data['text'].astype(str).tolist()
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts_train)

with open(METRICS_PATH, "rb") as f:
    metrics = pickle.load(f)
class_names = metrics["class_names"]

# -------------------------------
# LOAD CNN MODEL
# -------------------------------
print("🔹 Loading CNN emotion detection model...")
cnn_model = load_model(CNN_MODEL_PATH)
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE)
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# -------------------------------
# UTILS
# -------------------------------
def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s,.!?']", "", text)
    return text.lower()

def predict_bilstm(text: str):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    pred = bilstm_model.predict(padded, verbose=0)
    idx = int(np.argmax(pred))
    confidence = float(np.max(pred))
    return class_names[idx], confidence

def predict_cnn(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    results = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48))
        roi = roi_gray.astype("float")/255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)
        preds = cnn_model.predict(roi, verbose=0)
        label_idx = int(np.argmax(preds))
        label = emotion_labels[label_idx]
        confidence = float(np.max(preds))
        results.append({"label": label, "confidence": confidence, "bbox":[int(x),int(y),int(w),int(h)]})
    return results

# -------------------------------
# FLASK APP
# -------------------------------
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "<h3>Emotion Detection API (BiLSTM + CNN) Running ✅</h3>"

# Text prediction endpoint
@app.route("/predict", methods=["POST"])
def predict_text():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error":"No text provided"}), 400
    emotion, confidence = predict_bilstm(text)
    return jsonify({"model":"BiLSTM", "emotion":emotion, "confidence": round(confidence,3)})

# Camera feed for frontend
@app.route("/camera_feed")
def camera_feed():
    def generate():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            results = predict_cnn(frame)
            for res in results:
                x,y,w,h = res['bbox']
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, f"{res['label']} {res['confidence']:.2f}", (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Combined overall emotion
@app.route("/overall_emotion", methods=["POST"])
def overall_emotion():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    # ----- BiLSTM (Text) -----
    text_emotion, text_conf = None, 0.0
    if text:
        text_emotion, text_conf = predict_bilstm(text)

    # ----- CNN (Camera) -----
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    cnn_emotion, cnn_conf = None, 0.0
    if ret:
        cnn_results = predict_cnn(frame)
        if cnn_results:
            top = max(cnn_results, key=lambda x: x["confidence"])
            cnn_emotion, cnn_conf = top["label"], top["confidence"]

    # ----- Weighted Late Fusion -----
    TEXT_WEIGHT = 0.6
    CNN_WEIGHT  = 0.4
    all_emotions = set(emotion_labels + class_names)
    scores = {e: 0.0 for e in all_emotions}

    if text_emotion:
        scores[text_emotion] += TEXT_WEIGHT * text_conf
    if cnn_emotion:
        scores[cnn_emotion] += CNN_WEIGHT * cnn_conf

    # Pick highest weighted score
    overall_emotion = max(scores, key=scores.get)
    overall_conf = scores[overall_emotion]

    return jsonify({
        "text_emotion": text_emotion,
        "text_confidence": round(text_conf, 3) if text_conf else None,
        "camera_emotion": cnn_emotion,
        "camera_confidence": round(cnn_conf, 3) if cnn_conf else None,
        "overall_emotion": overall_emotion,
        "overall_confidence": round(overall_conf, 3)
    })


if __name__ == "__main__":
    print("🚀 Flask API running on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", debug=True, port=5000)
