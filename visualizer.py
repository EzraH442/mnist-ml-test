from loading_data import *


class shape_of_data:
    def print_shape_of_data():
        print("X_train: " + str(x_train.shape))
        print("Y_train: " + str(y_train.shape))
        print("X_test:  " + str(x_train.shape))
        print("Y_test:  " + str(y_train.shape))

    def graph_data():
        for i in range(1, 16):
            pyplot.subplot(3, 5, i)
            pyplot.imshow(x_train[i])
        pyplot.show()

    def call_function(function_to_call):
        if function_to_call == 'g':
            shape_of_data.graph_data()
        elif function_to_call == 's':
            shape_of_data.print_shape_of_data()
 
data = shape_of_data
wanted_function = input("Do you want The shape(s) or the graph(g):\n")
data.call_function(wanted_function)
