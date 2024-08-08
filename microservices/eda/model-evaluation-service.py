import logging
import json
import pika
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from flask_cors import CORS  
from threading import Event, Thread

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Store preprocessed data and evaluation results
preprocessed_data = None
model = None
evaluation_results = {}
evaluation_done = Event()

class ModelEvaluation:
    def __init__(self, model):
        self.model = model

    def evaluate(self, x_test, y_test):
        acc = self.model.evaluate(x_test, y_test)
        logging.info(f"Test set loss: {acc[0]}")
        logging.info(f"Test set accuracy: {acc[1] * 100:.2f}%")
        return acc

def process_preprocessed_data(ch, method, properties, body):
    global preprocessed_data
    logging.info("Received preprocessed data.")
    preprocessed_data = json.loads(body)
    # Acknowledge message receipt
    ch.basic_ack(delivery_tag=method.delivery_tag)
    # Check if we have both pieces of data
    if model:
        perform_evaluation()

def evaluate_model(ch, method, properties, body):
    global model
    logging.info("Received trained model for evaluation.")
    data = json.loads(body)
    model_path = data['model_path']
    logging.info(f"Loading model from: {model_path}")
    model = load_model(model_path)
    # Acknowledge message receipt
    ch.basic_ack(delivery_tag=method.delivery_tag)
    # Check if we have both pieces of data
    if preprocessed_data:
        perform_evaluation()

def perform_evaluation():
    global evaluation_results, preprocessed_data, model, evaluation_done
    try:
        if not preprocessed_data or not model:
            logging.error("Preprocessed data or model is missing.")
            return
        
        # Load preprocessed data
        x_test = preprocessed_data['x_test']
        y_test = preprocessed_data['y_test']
        
        # Create evaluation instance and perform evaluation
        evaluator = ModelEvaluation(model)
        results = evaluator.evaluate(x_test, y_test)
        
        # Store evaluation results
        evaluation_results = {
            'loss': results[0],
            'accuracy': results[1] * 100
        }
        
        # Log results
        logging.info(f"Model evaluation completed. Final Loss: {results[0]}, Final Accuracy: {results[1] * 100:.2f}%")
        
        # Set the event to indicate that evaluation is done
        evaluation_done.set()
    
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")

@app.route('/start_evaluation', methods=['POST'])
def start_evaluation():
    logging.info("Received request to start evaluation.")
    evaluation_done.clear()
    
    # Wait for evaluation to be done
    evaluation_done.wait()
    
    # Return results
    if evaluation_results:
        return jsonify(evaluation_results)
    else:
        return jsonify({"error": "Evaluation could not be completed."}), 500

@app.route('/evaluation_results', methods=['GET'])
def get_evaluation_results():
    logging.info("Fetching evaluation results.")
    return jsonify(evaluation_results)

def start_rabbitmq_consumers():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    
    # Declare queues
    channel.queue_declare(queue='trained_model_queue')
    channel.queue_declare(queue='preprocessed_data_queue')
    
    # Setup consumers
    channel.basic_consume(queue='trained_model_queue', on_message_callback=evaluate_model, auto_ack=False)
    channel.basic_consume(queue='preprocessed_data_queue', on_message_callback=process_preprocessed_data, auto_ack=False)
    
    logging.info("Evaluation service is now listening for trained models and preprocessed data.")
    channel.start_consuming()

if __name__ == '__main__':
    # Start RabbitMQ consumers in a separate thread
    thread = Thread(target=start_rabbitmq_consumers)
    thread.start()
    
    # Run Flask app
    app.run(debug=True, port=5003)
