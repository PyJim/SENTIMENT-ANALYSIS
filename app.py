from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')  # Load the fitted vectorizer

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Check if 'text' key is present in the JSON payload
    if 'text' not in data:
        return jsonify({'error': 'Missing \'text\' key in JSON payload'})

    text = data['text']

    # Transform the input text using the loaded vectorizer
    vectorized_text = vectorizer.transform([text])

    # Make predictions using the loaded model
    prediction = model.predict(vectorized_text)[0]

    # Convert the result to a string before returning it in the JSON response
    sentiment = str(prediction)

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run()
