import os
import numpy as np
import time
import cv2
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from deepface import DeepFace
import keras

# Fixed label dictionary (update based on your training labels)
labels = {
    0: 'Happy',
    1: 'Sad',
    2: 'Angry'
}

# Load the trained model
model = load_model('trained_model.h5')

# Start webcam
vc = cv2.VideoCapture(0)
time.sleep(2)

if not vc.isOpened():
    print("Cannot open webcam")
    exit()

print("Starting live detection. Press 'q' to quit.")

while True:
    ret, frame = vc.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize and preprocess image for model prediction
    resized_img = cv2.resize(frame, (300, 300))
    norm_img = img_to_array(resized_img) / 255.0
    norm_img = np.expand_dims(norm_img, axis=0)

    # Predict class using model
    preds = model.predict(norm_img, verbose=0)
    class_index = np.argmax(preds[0])
    predicted_class = labels.get(class_index, "Unknown")

    # DeepFace analysis
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emt = result[0]['dominant_emotion']
        region = result[0]['region']

        # Draw bounding box
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, f"{emt}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    except:
        emt = "unknown"

    # Status based on emotion
    if emt == "happy":
        status = "Nondepressed"
    elif emt == "neutral":
        status = "Depressed"
    else:
        status = "Uncertain"

    # Display information
    cv2.putText(frame, f"Class: {predicted_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Emotion: {emt}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f"Status: {status}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Live Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()
