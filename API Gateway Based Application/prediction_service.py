from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import logging
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the URL for the training microservice
TRAIN_MICROSERVICE_URL = 'http://localhost:5002/get_model_path'

# Hard-code class names for CIFAR-10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info("Starting prediction process.")
        
        # Get request data
        data = request.get_json()
        image_url = data['image_url']

        # Fetch the image from the URL
        logging.info("Fetching image from URL: %s", image_url)
        response = requests.get(image_url)
        if response.status_code != 200:
            logging.error("Error fetching image: %s", response.text)
            return jsonify({"error": "Error fetching image from URL."}), 500
        
        image = Image.open(BytesIO(response.content))
        image = image.convert('RGB')
        
        # Resize image to match the model input size (e.g., 32x32 for CIFAR-10)
        image = image.resize((32, 32))
        image_data = np.array(image)
        
        # Request model path from the training microservice
        logging.info("Requesting model path from the training microservice.")
        response = requests.get(TRAIN_MICROSERVICE_URL)
        if response.status_code != 200:
            logging.error("Error fetching model path from training microservice: %s", response.text)
            return jsonify({"error": "Error fetching model path from training microservice."}), 500
        
        model_path = response.json().get('model_path')
        if not model_path:
            logging.error("Model path not found.")
            return jsonify({"error": "Model path not found."}), 500
        
        # Load the model
        logging.info("Loading model from path: %s", model_path)
        model = load_model(model_path)
        logging.info("Model successfully loaded.")
        
        # Preprocess the image data
        logging.info("Preprocessing image data.")
        image_data = image_data.astype('float32') / 255.0
        image_data = np.expand_dims(image_data, axis=0)
        
        # Make prediction
        logging.info("Making prediction.")
        prediction = model.predict(image_data)
        class_idx = np.argmax(prediction, axis=1)[0]
        
        # Convert class index to class name
        class_name = CLASS_NAMES[class_idx]
        
        logging.info("Prediction completed. Class name: %s", class_name)
        
        return jsonify({"prediction": class_name})
    
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5004, debug=True)
