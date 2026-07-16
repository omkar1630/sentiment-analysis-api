import os
import pickle
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the saved Vectorizer and Model
# Ensure vector.pkl and sentiment.pkl are in the same directory as app.py
with open("vector.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("sentiment.pkl", "rb") as f:
    model = pickle.load(f)

# HTML Template for a simple user interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; line-height: 1.6; }
        textarea { width: 100%; height: 100px; padding: 10px; margin-bottom: 10px; border-radius: 5px; border: 1px solid #ccc; }
        input[type="submit"] { background-color: #007bff; color: white; padding: 10px 15px; border: none; border-radius: 5px; cursor: pointer; }
        input[type="submit"]:hover { background-color: #0056b3; }
        .result { margin-top: 20px; padding: 15px; border-radius: 5px; font-weight: bold; }
        .positive { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .negative { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    </style>
</head>
<body>
    <h2>Review Sentiment Predictor</h2>
    <form method="POST" action="/predict_html">
        <textarea name="review" placeholder="Enter your review here..." required>{{ review }}</textarea>
        <br>
        <input type="submit" value="Analyze Sentiment">
    </form>
    
    {% if sentiment %}
    <div class="result {{ sentiment }}">
        Predicted Sentiment: {{ sentiment.upper() }}
    </div>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE, review="", sentiment="")

# 1. UI Route (For testing directly in a browser)
@app.route('/predict_html', methods=['POST'])
def predict_html():
    review = request.form.get('review', '')
    if not review:
        return render_template_string(HTML_TEMPLATE, review="", sentiment="")
    
    # Transform the text using the loaded vectorizer
    transformed_text = vectorizer.transform([review])
    # Make prediction
    prediction = model.predict(transformed_text)[0]
    
    return render_template_string(HTML_TEMPLATE, review=review, sentiment=str(prediction))

# 2. API Route (For mobile apps or programmatic access via JSON)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if not data or 'review' not in data:
        return jsonify({"error": "Missing 'review' key in JSON body"}), 400
    
    review = data['review']
    transformed_text = vectorizer.transform([review])
    prediction = model.predict(transformed_text)[0]
    
    return jsonify({
        "review": review,
        "sentiment": str(prediction)
    })

if __name__ == '__main__':
    # Render binds to the environment variable PORT, falling back to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
