import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

def to_categorical(y, num_classes):
    """
    Converts a class vector (integers) to binary class matrix.
    
    Args:
    - y: numpy array of class labels (integers)
    - num_classes: total number of classes
    
    Returns:
    - numpy array: one-hot encoded matrix representation of y
    """
    y = np.asarray(y, dtype='int32')
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

class Cifar10Loading:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.x_val = None
        self.y_val = None

    def getShape(self):
        print("Training data shape:", self.x_train.shape)
        print("Test data shape:", self.x_test.shape)

class DataVisualization:
    @staticmethod
    def get_samples(x_train):
        fig, axes = plt.subplots(1, 5)
        for i in range(5):
            axes[i].imshow(x_train[i])
            axes[i].axis('off')
        plt.show()

    @staticmethod
    def classDistribution(data):
        sns.countplot(data.y_train.flatten())
        plt.show()

    @staticmethod
    def getMissingValues(data):
        print("Missing values in x_train:", data.x_train[data.x_train == None].size)
        print("Missing values in y_train:", data.y_train[data.y_train == None].size)
        print("Missing values in x_test:", data.x_test[data.x_test == None].size)
        print("Missing values in y_test:", data.y_test[data.y_test == None].size)

class DataPreprocessing:
    @staticmethod
    def normalizeX_Data(data):
        data.x_train = data.x_train.astype('float32') / 255.0
        data.x_test = data.x_test.astype('float32') / 255.0
        data.x_val = data.x_val.astype('float32') / 255.0

    @staticmethod
    def setValidationData(data, split_size=0.5):
        data.x_val = data.x_train[:int(split_size * len(data.x_train))]
        data.y_val = data.y_train[:int(split_size * len(data.y_train))]

class DataStatistics:
    @staticmethod
    def serialize_image(image):
        return image.tobytes()

    @staticmethod
    def to_serialized_dataframe(data, x, y):
        y = y.flatten()
        labels = [data.class_names[label] for label in y]
        serialized_images = [DataStatistics.serialize_image(img) for img in x]
        df = pd.DataFrame({'image': serialized_images, 'label': labels})
        return df

class CNNModel:
    def __init__(self, input_shape, weight_decay=1e-4):
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

        self.model = Model(inputs=inputs, outputs=outputs)

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, datagen, x_train, y_train, x_val, y_val, epochs=1):
        history = self.model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=epochs,
                                 steps_per_epoch=len(x_train) // 64, validation_data=(x_val, y_val),
                                 verbose=1)
        return history

    def save(self, filename):
        self.model.save(filename)

class ModelEvaluation:
    def __init__(self, model):
        self.model = model

    def evaluate(self, x_test, y_test):
        acc = self.model.evaluate(x_test, y_test)
        print("Test set loss:", acc[0])
        print("Test set accuracy:", acc[1] * 100)
        return acc

class ModelPlotting:
    @staticmethod
    def plot_results(history, epochs):
        epoch_range = range(1, epochs + 1)
        
        plt.plot(epoch_range, history.history['accuracy'])
        plt.plot(epoch_range, history.history['val_accuracy'])
        plt.title('Classification Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='lower right')
        plt.savefig("Classification Accuracy")
        plt.show()

        plt.plot(epoch_range, history.history['loss'])
        plt.plot(epoch_range, history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='lower right')
        plt.savefig("Model loss")
        plt.show()

def main():
    # Load and preprocess data
    data = Cifar10Loading()
    data.getShape()

    visualization = DataVisualization()
    visualization.get_samples(data.x_train)
    visualization.classDistribution(data)
    visualization.getMissingValues(data)

    DataPreprocessing.setValidationData(data, split_size=0.5)
    DataPreprocessing.normalizeX_Data(data)

    # Convert to DataFrame for TFDV
    train_df = DataStatistics.to_serialized_dataframe(data, data.x_train, data.y_train)

    # One-hot encode labels
    data.y_train = to_categorical(data.y_train, 10)
    data.y_test = to_categorical(data.y_test, 10)
    data.y_val = to_categorical(data.y_val, 10)

    # Data augmentation
    datagen = ImageDataGenerator()
    datagen.fit(data.x_train)

    print(data.x_train.shape)
    print(data.y_train.shape)
    print(data.y_test.shape)
    print(data.y_val.shape)

    # Model building and training
    cnn_model = CNNModel(input_shape=(32, 32, 3))
    cnn_model.compile()
    history = cnn_model.train(datagen, data.x_train, data.y_train, data.x_val, data.y_val, epochs=200)

    # Evaluate the model
    model_evaluation = ModelEvaluation(cnn_model.model)
    model_evaluation.evaluate(data.x_test, data.y_test)

    # Plot results
    ModelPlotting.plot_results(history, epochs=200)

    # Save the model
    cnn_model.save("CNN-91.74.h5")

if __name__ == "__main__":
    main()
