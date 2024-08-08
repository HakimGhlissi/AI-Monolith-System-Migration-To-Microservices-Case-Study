import json
import pika
import logging
from flask import Flask, jsonify
from flask_cors import CORS  
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import numpy as np
# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cnn_model(input_shape, weight_decay=1e-4):
    logging.info("Building CNN model.")
    inputs = Input(shape=input_shape)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)

    logging.info("CNN model built successfully.")
    return Model(inputs=inputs, outputs=outputs)

def train_model(ch, method, properties, body):
    logging.info("Received preprocessed data from the queue.")
    data = json.loads(body)
    
    # One-hot encode labels
    data['y_train'] = to_categorical(data['y_train'], 10)
    data['y_test'] = to_categorical(data['y_test'], 10)
    data['y_val'] = to_categorical(data['y_val'], 10)
    
    logging.info("Starting model training.")
    model = cnn_model(input_shape=(32, 32, 3))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(np.array(data['x_train']), np.array(data['y_train']), epochs=1, validation_data=(np.array(data['x_val']), np.array(data['y_val']))) #epochs is at 1 just to test the pipeline but can be adjusted for more for a better model training
    
    model.save('cnn_model.h5')
    logging.info("Model training completed and saved to 'cnn_model.h5'.")

    # Publish the trained model path to RabbitMQ
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='trained_model_queue')
    
    channel.basic_publish(exchange='',
                          routing_key='trained_model_queue',
                          body=json.dumps({'model_path': 'cnn_model.h5'}))
    
    connection.close()
    logging.info("Trained model path sent to 'trained_model_queue'.")
    return jsonify({"message": "Trained model path sent to 'trained_model_queue'."})


@app.route('/start_training', methods=['POST'])
def start_training():
    logging.info("Training service is starting and listening for preprocessed data.")
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='preprocessed_data_queue')
    
    channel.basic_consume(queue='preprocessed_data_queue', on_message_callback=train_model, auto_ack=True)
    
    channel.start_consuming()
    
    return jsonify({"message": "Training service is listening for preprocessed data."})

if __name__ == '__main__':
    app.run(debug=True, port=5002)
