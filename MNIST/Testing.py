"""
Benjamin Raiford — June 2020
Convolutional Neural Network for MNIST
    Provide a baseline model and a test harness for all models

This project owes a substantial debt to Jason Brownlee's tutorial at:
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
"""

# General Imports
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from numpy import mean
from numpy import std

# Keras Imports
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.utils import plot_model

"""
The MNIST Dataset can be found at:
http://yann.lecun.com/exdb/mnist/

The dataset is already split into two sets:
    A training set with 60,000 examples
    A testing set with 10,000 examples
    
    At each index of X is a different 28x28 image
    At each corresponding index in Y is the label for that image [0 through 9, inclusive]
    
    Each image is grayscale (that is, only one color channel)
    Each pixel has value in the range [0,255]
"""


# Load and reshape the datasets
def load_datasets():
    (trainX, trainY), (testX, testY) = mnist.load_data()

    """
    # DEBUG: Print shape of each set
    print('DEBUG: PRINTING SHAPES')
    print('\tTrain: X = %s, Y = %s' % (trainX.shape, trainY.shape))
    print('\tTest:  X = %s, Y = %s' % (testX.shape, testY.shape))
    """

    # Reshape the X values -- size of the dataset, image height, image width, number of color channels
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    # Reshape the Y values -- use one-hot encoding to assign categorical variables [0 through 9, inclusive]
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    return trainX, trainY, testX, testY


# Normalize the pixel range
def normalize_pixels(train, test):
    # Convert type from integer to float for normalization
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    # Normalize the pixel range: changing from [0,255] to [0,1]
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    # Return normalized images
    return train_norm, test_norm


# Build the CNN model
def build_baseline():
    model = Sequential()

    # CONSTRUCT LAYERS
    # Convolutional layer: 32 3x3 filters, ReLU activation function, He initializer
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
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


# Evaluate a model on label accuracy and loss
# We are passing the function itself, NOT building a model.
def evaluate_model(dataX, dataY, model_type, num_folds=6):
    """
    Why use validation data?
    https://stackoverflow.com/questions/46308374/what-is-validation-data-used-for-in-a-keras-sequential-model

    KFold Documentation:
        Overview — https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
        In depth — https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation

    General Algorithm:
        Pseudo-randomly shuffle the indices of the data
        Split the data into k groups (using indices)
        For each group i:
            sets aside group i as a test set
            trains on the remaining k-1 groups as one training dataset
            fits a model on these groups as one training set
            retains evaluation score and discards model

    Implementation notes:
        dataX is shuffled (using indices) and split into num_folds groups
        train_index is an array of indices (Ex: [1 4 5 ... 59996 59998 59999])
        test_index is an array of all indices not included in train_index (Ex: [2 3 6 ... 59995 59997 60000])
        NOTE: len(train_index) + len(test_index) == len(dataX)
        NOTE: Each fold has a different train_index and test_index
    """

    scores, histories = list(), list()
    kf = KFold(num_folds, shuffle=True, random_state=1)

    # For set of indices
    for train_index, test_index in kf.split(dataX):
        # Define model
        model = model_type()
        # Create train and test sets based off of indices
        trainX, trainY = dataX[train_index], dataY[train_index]
        testX, testY = dataX[test_index], dataY[test_index]

        # Fit model on training data
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)

        # Evaluate model on validation ("test") data
        _, accuracy = model.evaluate(testX, testY, verbose=0)

        # Append scores
        scores.append(accuracy)
        histories.append(history)
        print('Run', len(scores), 'accuracy is > %.3f' % (accuracy * 100.0))

    print('Accuracy: \n\tmean=%.3f \n\tstd=%.3f \n\tn=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))

    return scores, histories


# Plot learning curves for an evaluated model
def plot_evaluation(histories):
    print('What is the name of this model?')
    model_name = input()
    for i in range(len(histories)):
        # FIXME: https://stackoverflow.com/questions/46933824/matplotlib-adding-an-axes-using-the-same-arguments-as-a
        #  -previous-axes Plot Loss – Training set in blue, validation set in orange
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')

        # Plot Accuracy — Training set in blue, validation set in orange
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
        plt.tight_layout()

        # NOTE: loss noticeably below val_loss is a sign of over-fitting... the weights are less likely to generalize

    # NOTE: You must choose either show() or savefig() as both reset the current figure
    # plt.show()
    plt.savefig('%s_test.png' % model_name)


# Combine the functions to effectively evaluate a function
# We are passing the function itself, NOT building a model
def evaluation_harness(model_type):
    print("Evaluating", model_type.__name__)
    # Load dataset
    trainX, trainY, testX, testY = load_datasets()
    # Normalize pixels
    trainX, testX = normalize_pixels(trainX, testX)
    # Evaluate Model
    scores, histories = evaluate_model(trainX, trainY, model_type)
    # Plot learning curves
    plot_evaluation(histories)


# Train a model, test it on the test set, then save the weights
# We are passing the function itself, NOT building a model.
# Can comment out the evaluate lines (216-17) to save time
def final_test(model_type, save_name):
    print("Testing and saving", model_type.__name__)
    # Build the model
    model = model_type()
    # Load Dataset
    trainX, trainY, testX, testY = load_datasets()
    # Normalize Pixels
    trainX, testX = normalize_pixels(trainX, testX)
    # Fit model
    model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
    # Evaluate model on test dataset
    _, accuracy = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (accuracy * 100.0))
    # Save model
    model.save('%s.h5' % save_name)
