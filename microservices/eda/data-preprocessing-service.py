import json
import pika
import numpy as np
import logging
from flask import Flask, jsonify, request
from tensorflow.keras.datasets import cifar10
from flask_cors import CORS  

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoadingAndPreprocessing:
    def __init__(self):
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = None, None, None, None, None, None
    
    def load_data(self, sample_size=0.009): 
        #We worked with a small sample here because of the memory limit we had with RabbitMQ and to make testing phase and debugging faster but it can be adjusted to fit the entire dataset if needed as well
        
        logging.info("Loading CIFAR-10 data.")
        (x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()
        
        # Limit data to a sample of 20%
        limit_train = int(sample_size * len(x_train_full))
        limit_test = int(sample_size * len(x_test_full))
        
        self.x_train, self.y_train = x_train_full[:limit_train], y_train_full[:limit_train]
        self.x_test, self.y_test = x_test_full[:limit_test], y_test_full[:limit_test]
        
        logging.info(f"Data loaded with {len(self.x_train)} training samples and {len(self.x_test)} test samples.")
        return {"message": "Data loaded successfully."}

    def preprocess_data(self, split_size=0.5):
        logging.info("Starting data preprocessing.")
        
        self.x_val = self.x_train[:int(split_size * len(self.x_train))]
        self.y_val = self.y_train[:int(split_size * len(self.y_train))]
        
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0
        self.x_val = self.x_val.astype('float32') / 255.0
        
        logging.info("Data preprocessing completed.")
        return {"message": "Data preprocessed successfully."}

data_service = DataLoadingAndPreprocessing()

@app.route('/load_and_preprocess', methods=['POST'])
def load_and_preprocess():
    try:
        data_service.load_data()
        data_service.preprocess_data()
        
        connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        channel = connection.channel()
        
        channel.queue_declare(queue='preprocessed_data_queue')
        
        data_message = {
            'x_train': data_service.x_train.tolist(),
            'y_train': data_service.y_train.tolist(),
            'x_test': data_service.x_test.tolist(),
            'y_test': data_service.y_test.tolist(),
            'x_val': data_service.x_val.tolist(),
            'y_val': data_service.y_val.tolist(),
            'class_names': data_service.class_names
        }
        
        logging.info("Publishing preprocessed data to RabbitMQ queue.")
        
        channel.basic_publish(exchange='', routing_key='preprocessed_data_queue', body=json.dumps(data_message))
        
        connection.close()
        
        logging.info("Data loaded, preprocessed, and sent to the queue successfully.")
        
        return jsonify({"message": "Data loaded, preprocessed, and sent to the queue successfully."})
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"message": "An error occurred."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
