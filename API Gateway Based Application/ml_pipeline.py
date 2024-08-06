import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical, plot_model
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# import tensorflow_data_validation as tfdv
import pandas as pd

class Cifar10Loading:
    def __init__(self, sample_fraction=1.0):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.x_val = None
        self.y_val = None

        if sample_fraction < 1.0:
            self.x_train, self.y_train = self.sample_data(self.x_train, self.y_train, sample_fraction)
            self.x_test, self.y_test = self.sample_data(self.x_test, self.y_test, sample_fraction)

    def sample_data(self, x, y, fraction):
        num_samples = int(len(x) * fraction)
        indices = np.random.choice(len(x), num_samples, replace=False)
        return x[indices], y[indices]

    def getShape(self):
        print("Training data shape:", self.x_train.shape)
        print("Test data shape:", self.x_test.shape)

    def to_json(self):
        return {
            "x_train": self.x_train.tolist(),
            "y_train": self.y_train.tolist(),
            "x_test": self.x_test.tolist(),
            "y_test": self.y_test.tolist(),
            "x_val": self.x_val.tolist() if self.x_val is not None else None,
            "y_val": self.y_val.tolist() if self.y_val is not None else None,
            "class_names": self.class_names,
        }

    @staticmethod
    def from_json(json_data):
        obj = Cifar10Loading.__new__(Cifar10Loading)
        obj.x_train = np.array(json_data["x_train"])
        obj.y_train = np.array(json_data["y_train"])
        obj.x_test = np.array(json_data["x_test"])
        obj.y_test = np.array(json_data["y_test"])
        obj.x_val = np.array(json_data["x_val"]) if json_data["x_val"] is not None else None
        obj.y_val = np.array(json_data["y_val"]) if json_data["y_val"] is not None else None
        obj.class_names = json_data["class_names"]
        return obj

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

    # @staticmethod
    # def generate_statistics(train_df):
    #     stats_options = tfdv.StatsOptions(enable_semantic_domain_stats=True)
    #     train_stats = tfdv.generate_statistics_from_dataframe(train_df, stats_options)
    #     tfdv.visualize_statistics(train_stats)
    #     schema = tfdv.infer_schema(train_stats)
    #     tfdv.display_schema(schema)
    #     return train_stats, schema


class CNNModel:
    def __init__(self, input_shape, weight_decay=1e-4):
        self.model = Sequential([
            Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=input_shape),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.2),

            Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),

            Conv2D(256, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.4),

            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])

    def compile(self):
        opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, datagen, x_train, y_train, x_val, y_val, epochs=20):
        steps_per_epoch = len(x_train) // 64  # Use integer division
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, cooldown=1, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=15)
        ]
        history = self.model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=epochs,
                                 steps_per_epoch=steps_per_epoch, validation_data=(x_val, y_val),
                                 verbose=1, callbacks=callbacks)
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

    # Generate statistics with TFDV
    train_stats, schema = DataStatistics.generate_statistics(train_df)

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
    history = cnn_model.train(datagen, data.x_train, data.y_train, data.x_val, data.y_val, epochs=20)

    # Evaluate the model
    model_evaluation = ModelEvaluation(cnn_model.model)
    model_evaluation.evaluate(data.x_test, data.y_test)

    # Plot results
    ModelPlotting.plot_results(history, epochs=20)

    # Save the model
    cnn_model.save("CNN-91.74.h5")

if __name__ == "__main__":
    main()
