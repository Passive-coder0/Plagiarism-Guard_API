import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import AIDetector

app = Flask(__name__)
CORS(app)

detector = AIDetector()

# Add a test route
@app.route('/')
def home():
    return jsonify({"message": "AI Detector API is running!"})

@app.route('/detect', methods=['POST'])
def detect_text():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        result = detector.detect(text)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = os.environ.get("PORT", 5000)
    app.run(host='0.0.0.0', port=port)