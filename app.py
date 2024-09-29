from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)

# Load model and tokenizer
model = tf.keras.models.load_model('emotion_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

EMOTION_MAPPING = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

def predict_emotion(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    prediction = model.predict(padded)
    predicted_emotion = EMOTION_MAPPING[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return predicted_emotion, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    emotion, confidence = predict_emotion(text)
    return jsonify({'emotion': emotion, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)