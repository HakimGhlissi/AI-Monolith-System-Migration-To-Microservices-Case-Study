from flask import Flask, jsonify
from flask_cors import CORS  # Import CORS
from ml_pipeline import CNNModel
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import logging
import requests
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS

PREPROCESSING_MICROSERVICE_URL = 'http://localhost:5001/get_preprocessed_data'

@app.route('/train', methods=['POST'])
def train_model():
    try:
        logging.info("Starting model training process.")
        
        # Fetch preprocessed data from the first microservice
        logging.info("Requesting preprocessed data from microservice.")
        response = requests.get(PREPROCESSING_MICROSERVICE_URL)
        if response.status_code != 200:
            logging.error("Error fetching preprocessed data: %s", response.text)
            return jsonify({"error": "Error fetching preprocessed data."}), 500
        
        # Parse the preprocessed data directly from the response
        data = response.json()
        logging.info("Retrieved preprocessed data from preprocessing microservice.")

        # Convert to NumPy arrays directly
        logging.info("Converting data to NumPy arrays.")
        x_train = np.array(data['x_train'])
        y_train = np.array(data['y_train'])
        x_val = np.array(data['x_val'])
        y_val = np.array(data['y_val'])

        # Ensure the arrays are correctly shaped and values are appropriate
        logging.info("Ensuring arrays are correctly shaped and values are appropriate.")
        x_train = np.array(x_train, dtype='float32')
        x_val = np.array(x_val, dtype='float32')
        y_train = np.array(y_train, dtype='int')
        y_val = np.array(y_val, dtype='int')

        # One-hot encode labels
        logging.info("One-hot encoding labels.")
        y_train = to_categorical(y_train, 10)
        y_val = to_categorical(y_val, 10)
        logging.info("Labels one-hot encoded.")

        # Data augmentation
        logging.info("Starting data augmentation.")
        datagen = ImageDataGenerator()
        datagen.fit(x_train)
        logging.info("Data augmentation completed.")

        # Model building and training
        logging.info("Building and compiling the model.")
        cnn_model = CNNModel(input_shape=(32, 32, 3))
        cnn_model.compile()
        logging.info("Model compiled.")

        # Ensure epochs is an integer
        epochs = 20
        logging.info("Starting model training with %d epochs.", epochs)
        history = cnn_model.train(datagen, x_train, y_train, x_val, y_val, epochs=epochs)
        logging.info("Model training completed.")

        # Save the model
        model_path = "CNN-91.74.h5"
        logging.info("Saving the trained model to %s", model_path)
        cnn_model.save(model_path)
        logging.info("Model saved to %s", model_path)

        return jsonify({"message": "Model trained and saved successfully.", "model_path": model_path})
    except Exception as e:
        logging.error("Error during model training: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/get_model_path', methods=['GET'])
def get_model_path():
    try:
        model_path = "CNN-91.74.h5"  # Adjust the path as necessary
        return jsonify({"model_path": model_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5002, debug=True)
