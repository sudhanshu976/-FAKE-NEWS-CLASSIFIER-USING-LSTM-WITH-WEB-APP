from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import re

app = Flask(__name__)

# Load the tokenizer and model
tokenizer = Tokenizer()  # Replace with the actual function to load your tokenizer
max_sequence_length = 42  # Replace with your actual max_sequence_length
model = load_model('models/news_classifier.h5')

# Preprocessing Logic
def preprocess_text(text):
    # Replace characters that are not between a to z or A to Z with whitespace
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Convert all characters into lower-case ones.
    text = text.lower()

    # Remove inflectional morphemes like "ed", "est", "s", and "ing" from their token stem
    text = re.sub('(ed|est|s|ing)$', '', text)
    return text


# Home route
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        title_input = request.form['title']

        # Preprocess the title
        preprocessed_title = preprocess_text(title_input)

        # Tokenize and pad the preprocessed title
        sequence = tokenizer.texts_to_sequences([preprocessed_title])
        padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)

        # Prediction
        prediction = model.predict(padded_sequence)[0]

        # Classify as Real or Fake
        result = "REAL NEWS" if prediction >= 0.5 else "FAKE NEWS"

        return render_template('index.html', title=title_input, result=result)


if __name__ == '__main__':
    app.run(debug=True)
