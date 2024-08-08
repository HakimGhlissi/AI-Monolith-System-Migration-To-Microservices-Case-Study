import logging
import io
import pika
import numpy as np
import requests
import tensorflow as tf
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS  
from threading import Event, Thread
import json

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dummy class names for prediction mapping
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Global model variable
model = None
model_loaded = Event()

def load_model_from_queue(ch, method, properties, body):
    global model
    model_path = json.loads(body)['model_path']
    
    try:
        # Load the model from the saved file
        model = load_model(model_path)
        logging.info(f"Model loaded from file {model_path}.")
        model_loaded.set()  # Signal that the model is loaded
    except Exception as e:
        logging.error(f"Failed to load model from file {model_path} - Error: {e}")
        model = None
        model_loaded.set()  # Signal failure
    finally:
        ch.basic_ack(delivery_tag=method.delivery_tag)

@app.route('/predict', methods=['POST'])
def predict():
    global model
    # Ensure model is loaded
    if model is None:
        logging.info("Model is not loaded, attempting to load.")
        if not model_loaded.is_set():
            # Load the model if not already loaded
            model_path = 'cnn_model.h5'  # Path to the model
            try:
                model = load_model(model_path)
                logging.info(f"Model loaded from file {model_path}.")
                model_loaded.set()
            except Exception as e:
                logging.error(f"Failed to load model from file {model_path} - Error: {e}")
                return jsonify({"error": "Model is not available."}), 503
    
    image_url = request.json.get('image_url', None)
    
    if image_url is None:
        logging.error("No image URL provided.")
        return jsonify({"error": "Image URL is required."}), 400
    
    logging.info(f"Received prediction request with image_url: {image_url}")
    
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        logging.info(f"Image fetched successfully from URL: {image_url}")
    except Exception as e:
        logging.error(f"Failed to fetch image from URL: {image_url} - Error: {e}")
        return jsonify({"error": "Failed to fetch image from URL."}), 400
    
    try:
        img = image.load_img(io.BytesIO(response.content), target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0
        
        logging.info("Image preprocessing completed. Making prediction.")
        
        # Prediction
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names[class_idx]
        
        logging.info(f"Prediction completed. Predicted class: {predicted_class}")
        return jsonify({"prediction": predicted_class})
    except Exception as e:
        logging.error(f"Error during prediction - Error: {e}")
        return jsonify({"error": "Error during prediction."}), 500

def start_rabbitmq_consumers():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    
    # Declare queues
    channel.queue_declare(queue='trained_model_queue')
    
    # Setup consumer for model queue
    channel.basic_consume(queue='trained_model_queue', on_message_callback=load_model_from_queue, auto_ack=False)
    
    logging.info("RabbitMQ consumer is now listening for models.")
    channel.start_consuming()

if __name__ == '__main__':
    # Start RabbitMQ consumers in a separate thread
    thread = Thread(target=start_rabbitmq_consumers)
    thread.start()
    
    # Run Flask app
    app.run(debug=True, port=5004)
