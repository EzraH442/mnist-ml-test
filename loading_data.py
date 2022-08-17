from keras.datasets import mnist
from matplotlib import pyplot
import keras


(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = 10

y_train_final = keras.utils.to_categorical(y_train, num_classes)
y_test_final = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train_final = x_train.reshape(x_train.shape[0], -1)
x_test_finale = x_test.reshape(x_test.shape[0], -1)
