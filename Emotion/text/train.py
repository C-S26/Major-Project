import numpy as np
import pandas as pd
import nltk
import pickle
nltk.download('punkt')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load Dataset
# -----------------------------
# CSV must have 2 columns: "text" and "label"
# Example row: "This game has pissed me off...", "anger"
data = pd.read_csv("isear.csv")

texts = data['text'].astype(str).tolist()
labels_raw = data['label'].astype(str).tolist()

# -----------------------------
# Encode Labels (string → int)
# -----------------------------
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels_raw)   # anger→0, joy→1, etc.
y = to_categorical(labels)

# -----------------------------
# Tokenization & Padding
# -----------------------------
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

maxlen = 100
X = pad_sequences(sequences, maxlen=maxlen, padding='post')

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Build BiLSTM Model
# -----------------------------
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=maxlen))
model.add(Bidirectional(LSTM(128, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(y.shape[1], activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# -----------------------------
# Training Loop with Metrics
# -----------------------------
epochs = 10
batch_size = 32

train_losses = []
train_accuracies = []
precision_list = []
recall_list = []
f1_list = []
epoch_labels_all = []
epoch_preds_all = []

for epoch in range(epochs):
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=1,
                        validation_data=(X_val, y_val), verbose=1)
    
    # Record metrics
    loss = history.history['loss'][0]
    acc = history.history['accuracy'][0]
    train_losses.append(loss)
    train_accuracies.append(acc)

    # Predictions on validation set
    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(y_val, axis=1)

    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    epoch_labels_all.append(y_true.tolist())
    epoch_preds_all.append(y_pred.tolist())

# -----------------------------
# Save Model
# -----------------------------
model.save("emotion_bilstm.h5")  # Saved Keras model
print("✅ Model saved as emotion_bilstm.h5")

# -----------------------------
# Save Metrics
# -----------------------------
metrics = {
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'precision_list': precision_list,
    'recall_list': recall_list,
    'f1_list': f1_list,
    'epoch_labels_all': epoch_labels_all,
    'epoch_preds_all': epoch_preds_all,
    'class_names': list(label_encoder.classes_)  # save original string labels
}

with open("metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print("✅ Metrics saved in metrics.pkl")
