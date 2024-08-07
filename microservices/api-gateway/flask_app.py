from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/load_and_preprocess', methods=['POST'])
def load_and_preprocess():
    try:
        response = requests.post('http://localhost:5001/load_and_preprocess')
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    data = request.get_json()
    print("log1")
    response = requests.post('http://localhost:5002/train', json=data)
    return jsonify(response.json())

@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    data = request.get_json()
    response = requests.post('http://localhost:5003/evaluate', json=data)
    return jsonify(response.json())

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    response = requests.post('http://localhost:5004/predict', json=data)
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(port=5000, debug=True)