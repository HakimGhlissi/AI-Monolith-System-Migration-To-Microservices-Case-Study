from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import requests
import logging
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the URLs for the training and preprocessing microservices
TRAIN_MICROSERVICE_URL = 'http://localhost:5002/get_model_path'
PREPROCESSING_MICROSERVICE_URL = 'http://localhost:5001/get_preprocessed_data'

@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    try:
        logging.info("Starting model evaluation process.")
        
        # Fetch preprocessed data from the preprocessing microservice
        logging.info("Requesting preprocessed data from the preprocessing microservice.")
        response = requests.get(PREPROCESSING_MICROSERVICE_URL)
        if response.status_code != 200:
            logging.error("Error fetching preprocessed data: %s", response.text)
            return jsonify({"error": "Error fetching preprocessed data."}), 500
        
        data = response.json()
        logging.info("Successfully retrieved preprocessed data.")
        
        # Check if 'x_test' and 'y_test' are in the response
        if 'x_test' not in data or 'y_test' not in data:
            logging.error("Preprocessed data missing 'x_test' or 'y_test'.")
            return jsonify({"error": "Preprocessed data is missing 'x_test' or 'y_test'."}), 500
        
        # Convert data to NumPy arrays
        logging.info("Converting test data to NumPy arrays.")
        x_test = np.array(data['x_test'])
        y_test = np.array(data['y_test'])
        
        # Log shapes of the arrays
        logging.info("x_test shape: %s", x_test.shape)
        logging.info("y_test shape: %s", y_test.shape)
        
        # Check if x_test and y_test have the same number of samples
        if x_test.shape[0] != y_test.shape[0]:
            logging.error("Mismatch in number of samples between x_test and y_test: %d vs %d", x_test.shape[0], y_test.shape[0])
            return jsonify({"error": "Mismatch in number of samples between x_test and y_test."}), 500
        
        # One-hot encode labels
        logging.info("One-hot encoding labels.")
        y_test = to_categorical(y_test, 10)
        
        # Evaluate the model
        logging.info("Requesting model path from the training microservice.")
        response = requests.get(TRAIN_MICROSERVICE_URL)
        if response.status_code != 200:
            logging.error("Error fetching model path from training microservice: %s", response.text)
            return jsonify({"error": "Error fetching model path from training microservice."}), 500
        
        model_path = response.json().get('model_path')
        if not model_path:
            logging.error("Model path not found.")
            return jsonify({"error": "Model path not found."}), 500
        
        logging.info("Loading model from path: %s", model_path)
        model = load_model(model_path)
        logging.info("Model successfully loaded.")
        
        logging.info("Evaluating the model.")
        acc = model.evaluate(x_test, y_test)
        
        logging.info("Model evaluation completed. Loss: %.4f, Accuracy: %.4f", acc[0], acc[1])
        
        return jsonify({"loss": acc[0], "accuracy": acc[1]})
    
    except Exception as e:
        logging.error("Error during model evaluation: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5003, debug=True)
