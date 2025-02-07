import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import AIDetector

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize detector
detector = AIDetector()

@app.route('/')
def home():
    return jsonify({"message": "API is running"})

@app.route('/detect', methods=['POST'])
def detect_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        result = detector.detect(data['text'])
        return jsonify(result)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Remove the if __name__ == '__main__' block completely