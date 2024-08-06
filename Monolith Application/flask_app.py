from flask import Flask, jsonify, request
from ml_pipeline import Cifar10Loading, DataPreprocessing, DataStatistics, CNNModel, ModelEvaluation, ModelPlotting
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO

# Define class names corresponding to CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/load_and_preprocess', methods=['POST'])
def load_and_preprocess():
    global data
    data = Cifar10Loading()
    DataPreprocessing.setValidationData(data, split_size=0.5)
    DataPreprocessing.normalizeX_Data(data)
    
    return jsonify({"message": "Data loaded and preprocessed successfully."})

@app.route('/train', methods=['POST'])
def train_model():
    # Check if data is loaded and preprocessed
    if 'data' not in globals():
        return jsonify({"error": "Data not loaded and preprocessed."}), 400
    
    global cnn_model, history
    # One-hot encode labels
    data.y_train = to_categorical(data.y_train, 10)
    data.y_test = to_categorical(data.y_test, 10)
    data.y_val = to_categorical(data.y_val, 10)

    print("message : Model will be trained")
    # Data augmentation
    datagen = ImageDataGenerator()
    datagen.fit(data.x_train)
    print("message : Data augmentation successfull")

    # Model building and training
    cnn_model = CNNModel(input_shape=(32, 32, 3))
    cnn_model.compile()
    print("message : Model compiled successfully")
    
    # Ensure epochs is an integer
    epochs = 1
    print(f"Epochs type before training:", epochs)  # Check epochs type
    history = cnn_model.train(datagen, data.x_train, data.y_train, data.x_val, data.y_val, epochs=epochs)

    # Save the model
    model_path = "CNN-91.74.h5"
    cnn_model.save("./CNN-91.74.h5")

    return jsonify({"message": "Model trained and saved successfully.", "model_path": model_path})

@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    if 'cnn_model' not in globals():
        return jsonify({"error": "Model not trained."}), 400
    
    model_path = request.json['model_path']
    if not os.path.exists(model_path):
        return jsonify({"error": "Model file not found."}), 404

    # Load the model
    model = load_model(model_path)
    
    # Evaluate the model
    model_evaluation = ModelEvaluation(model)
    acc = model_evaluation.evaluate(data.x_test, data.y_test)

    return jsonify({"loss": acc[0], "accuracy": acc[1]})

@app.route('/predict', methods=['POST'])
def predict():
    model_path = request.json['model_path']
    image_url = request.json['image_url']

    if not os.path.exists(model_path):
        return jsonify({"error": "Model file not found."}), 404

    # Load the model
    model = load_model(model_path)

    # Load and preprocess the image from URL
    response = requests.get(image_url)
    img = image.load_img(BytesIO(response.content), target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    predicted_class = class_names[class_idx]

    return jsonify({"prediction": predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
