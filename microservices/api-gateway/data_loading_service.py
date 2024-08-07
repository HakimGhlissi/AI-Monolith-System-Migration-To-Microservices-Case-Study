from flask import Flask, jsonify
from ml_pipeline import Cifar10Loading, DataPreprocessing
import logging
import numpy as np
from flask_cors import CORS  

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

preprocessed_data = None

@app.route('/load_and_preprocess', methods=['POST'])
def load_and_preprocess():
    global preprocessed_data
    try:
        # Hardcode sample_fraction to 0.2 to use 20% of the data
        # We used 20% to have a faster runtime when checking/validating the ml pipeline but it can adjsuted to 1.0 to use the entire dataset
        sample_fraction = 0.2

        logging.info("Starting data loading process.")
        data = Cifar10Loading(sample_fraction=sample_fraction)
        logging.info("Data loaded successfully.")

        logging.info("Starting data preprocessing.")
        DataPreprocessing.setValidationData(data, split_size=0.5)
        DataPreprocessing.normalizeX_Data(data)
        logging.info("Data preprocessed successfully.")

        # Store preprocessed data in memory
        preprocessed_data = {
            'x_train': data.x_train.tolist(),
            'y_train': data.y_train.tolist(),
            'x_val': data.x_val.tolist(),
            'y_val': data.y_val.tolist(),
            'x_test': data.x_test.tolist(),
            'y_test': data.y_test.tolist(),
            'class_names': data.class_names
        }

        return jsonify({"message": "Data loaded and preprocessed successfully."})
    except Exception as e:
        logging.error("Error during data loading or preprocessing: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/get_preprocessed_data', methods=['GET'])
def get_preprocessed_data():
    if preprocessed_data is None:
        return jsonify({"error": "Preprocessed data not available."}), 400
    return preprocessed_data

if __name__ == '__main__':
    app.run(port=5001, debug=True)
