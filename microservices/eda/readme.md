# Machine Learning Pipeline with Event-Driven Architecture (EDA)

## Project Context

This project is part of the research paper titled **"The Migration of Monolithic Systems to AI-Based Microservices Using AI Design Patterns"**. The primary aim of this work is to demonstrate the migration of AI-based systems from a monolithic to a microservice-based architecture using specific design patterns. As there were no available monolithic AI applications with a microservice-based version for validation, we created our own application to test our migration approach.

## Overview

The application is an image classification system using the CIFAR-10 dataset to classify various types of images. This implementation leverages an Event-Driven Architecture (EDA) to manage communication between different microservices, providing a more scalable and flexible solution compared to traditional monolithic systems.

## System Architecture

### Components

The system is comprised of several microservices, each responsible for a specific part of the machine learning pipeline. Here's a brief overview of each component:

1. **Data Loading and Preprocessing Microservice**
   - **Responsibilities**: Handles the loading and preprocessing of data, preparing it for model training.
   - **Communication**: Publishes preprocessed data to a RabbitMQ queue, which is consumed by the model training microservice.

2. **Model Training Microservice**
   - **Responsibilities**: Receives preprocessed data from the data loading microservice, trains the machine learning model, and then publishes the model's path to a RabbitMQ queue.
   - **Communication**: The model's path is sent to the evaluation microservice via RabbitMQ.

3. **Model Evaluation Microservice**
   - **Responsibilities**: Consumes the model's path and evaluates its performance. Results are then forwarded to the prediction microservice.
   - **Communication**: Results are sent through RabbitMQ for use by the prediction microservice.

4. **Prediction Microservice**
   - **Responsibilities**: Uses the trained model to make predictions on new images. It listens for new prediction requests and responds with the classification results.

### Event-Driven Architecture (EDA)

In our EDA-based system:

- **Events**: Events are used to communicate between microservices. Each service publishes events to RabbitMQ, which are then consumed by other services. This approach decouples the services and enables them to operate independently while still cooperating to complete the pipeline.

- **RabbitMQ**: Acts as the message broker, facilitating communication between microservices. Each microservice is set up to listen to specific queues for incoming messages and to publish messages to queues as needed. This setup ensures that data flows smoothly through the pipeline, from preprocessing to prediction.

### Architecture Diagram

![Architecture Diagram](static/MLPipelinemono.png)

## Setup and Installation

To get the application up and running, follow these steps:

1. **Install Dependencies**

   Create a virtual environment and install the necessary dependencies using the `requirements.txt` file:

   ```bash
   $ pip install -r requirements.txt
    ```

2. **Run RabbitMQ**
Ensure RabbitMQ is running on your local machine. You can use Docker too to start RabbitMQ with the following command:

    ```bash
    $ docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:management
    ```

3. **Start the Microservices**
Launch each microservice in separate terminal windows or tabs. For example, you might start them with the following commands:

    ```bash
    $ python data_preprocessing_service.py
    $ python model_training_service.py
    $ python model_evaluation_service.py
    $ python prediction_service.py
    ```

## Usage
Load and Preprocess Data: Initiate the data loading and preprocessing step by sending a request to the corresponding endpoint.
Train Model: Start the model training process by sending a request to the training endpoint.
Evaluate Model: Trigger the evaluation of the trained model by sending a request to the evaluation endpoint.
Predict: Make predictions using the trained model by sending an image URL to the prediction endpoint.

## Requirements
Python (version 3.6 or higher)
Flask: Web framework for creating the microservices
TensorFlow: Machine learning library
NumPy: Library for numerical operations
Pika: Python RabbitMQ client library
Requests: HTTP library for fetching images
Flask-CORS: Cross-Origin Resource Sharing support for Flask
RabbitMQ: Message broker for communication between microservices
