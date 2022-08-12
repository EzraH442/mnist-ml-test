import tensorflow as tf
from tensorflow import keras

class Model:
    def __init__(self, path = None):
        if path != None:
            self.model = keras.models.load_model(path)
            print("###### Loaded Model ######")
        else:
            num_classes = 10
            data_shape = (28, 28, 1)

            mnist_inputs = keras.Input(shape=data_shape)
            conv_first = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(mnist_inputs)
            maxpool_first = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_first)
            conv_second = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(maxpool_first)
            maxpool_second = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_second)
            # flat = keras.layers.Flatten()(maxpool_second)
            # dropout = keras.layers.Dropout(0.5)(flat)
            # dense = keras.layers.Dense(num_classes, activation="softmax")(dropout)
            dense = keras.layers.Dense(num_classes, activation="softmax")(maxpool_second)

            self.model = keras.Model(inputs=mnist_inputs, outputs=dense, name="mnist_model")
            print("###### Initialized Model ######")

        self.model.summary()

    def train(self, data):
        print("traing")

    def test(self, data):
        print("test")

    def save(self, path: str):
        self.model.save(path) 

