from MNIST.Testing import *

"""
This file provides an example of how you would create, test, and save a new model.
All that has been changed in build_new_model() is added two convolutional layers for increased depth.
This increased depth makes times a lot longer... if you want to make sure your program is running,
a good idea is to change verbose from 0 to 1 in Testing.py line 143 (only to check that it is running, this gives a ton
of output).
"""


def build_new_model():
    model = Sequential()

    # CONSTRUCT LAYERS
    # Convolutional layer: 32 3x3 filters, ReLU activation function, He initializer
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    # Max pooling layer with 2x2 filter
    model.add(MaxPooling2D((2, 2)))
    # Two convolutional layers, each with 64 3x3 filters, ReLU activation function, He initializer
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    # Max pooling layer with 2x2 filter
    model.add(MaxPooling2D((2, 2)))
    # Flatten filter maps to pass to classifier
    model.add(Flatten())
    # Fully-connected layer with 100 nodes, ReLU activation function, He initializer
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    # Fully-connected output layer with 10 nodes (for the labels [0,9])
    model.add(Dense(10, activation='softmax'))

    # DEFINE OPTIMIZER
    # Use a stochastic gradient descent optimizer with learning rate 0.01 and momentum 0.90
    # For notes on momentum see: https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
    opt = SGD(lr=0.01, momentum=0.9)

    # DEFINE LOSS FUNCTION
    # Observe that the loss function used is "categorical_crossentropy" which is appropriate for our categorical labels
    loss_func = 'categorical_crossentropy'

    # COMPILE MODEL
    model.compile(optimizer=opt, loss=loss_func, metrics=['accuracy'])

    return model


# Entry Point
# NOTE: Be sure to pass the function itself rather than calling a build function
# Evaluate new model
evaluation_harness(build_new_model)
# Test and save weights for new model
final_test(build_new_model, "new_model")
