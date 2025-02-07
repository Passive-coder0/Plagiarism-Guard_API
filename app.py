import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import AIDetector

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

detector = AIDetector()

# Test route
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API is running"})

# Main detection route
@app.route('/detect', methods=['POST'])  # Explicitly allow POST
def detect_text():
    try:
        # Print request data for debugging
        print("Received request:", request.method)
        print("Request data:", request.get_json())

        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        result = detector.detect(text)
        return jsonify(result)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)