# The Migration of Monolithic Systems to AI-Based Microservices Using AI Design Patterns

This repository contains the code and resources for the paper **"The Migration of Monolithic Systems to AI-Based Microservices Using AI Design Patterns."** The paper explores a guided approach for migrating AI monolithic systems into microservices using design patterns.

## Overview

In our paper, we discuss various design patterns that can be considered for the migration process and propose a systematic approach to guide this migration. Recognizing the lack of existing monolithic AI systems with corresponding microserviced versions, we developed our own set of applications to test and validate our approach.

## Contents

This repository includes:

1. **Monolithic System**: An image classification application that uses the CIFAR-10 dataset for training. It incorporates a Convolutional Neural Network (CNN) model and defines various parts of the ML pipeline within the monolithic application.

2. **Microservice-Based Systems**:
   - **API Gateway Design Pattern**: A microservice version utilizing the API Gateway pattern.
   - **Event-Driven Architecture (EDA) Design Pattern**: Another microservice version employing the EDA pattern.

Both microservice implementations adhere to the guidelines and process outlined in our paper.

## Structure

├── monolith   
│ │ ├── Static  
│ │ │ index.html   
│ ├── app.py   
│ ├── ml_pipeline.py #Model_Definition  
│ ├── CNN-91.74.h5 #Model_Used  
├── microservices  
│ ├── api-gateway  
│ ├── eda  
└── README.md


- **`monolith`**: Contains the monolithic image classification application.
- **`microservices`**: Contains the microservice-based implementations.
  - **`api-gateway`**: Microservice system using the API Gateway pattern.
  - **`eda`**: Microservice system using the Event-Driven Architecture pattern.

## Getting Started

### Prerequisites

- Python 3.x
- Additional dependencies listed in the respective `requirements.txt` files.

### Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
[Monolith Application]
2. **Install dependencies for the monolithic system**:
   ```sh
  cd monolith
   pip install -r requirements.txt
```
3. **Set up and run the monolithic application**:
   ```sh
   python app.py

### Usage 
  Access the monolithic application via http://localhost:5000.

### Acknowledgments
  The CIFAR-10 dataset, collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
