"""
Benjamin Raiford — June 2020
Convolutional Neural Network for MNIST
    Predict the class given an image

This project owes a substantial debt to Jason Brownlee's tutorial at:
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
"""

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
from os import listdir

from MNIST.Testing import load_datasets


# Load and Prepare Image
def load_image(filename):
    # load the image
    img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


# Load images from MNIST dataset
def load_testing():
    # load test dataset
    test_images, test_labels = load_datasets()[2:4]
    # prepare pixel data
    test_images = test_images.astype('float32')
    test_images = test_images / 255.0
    # ensure that we are feeding shape = (1, 28, 28, 1) into model.predict()
    test_images = test_images[:, np.newaxis, ...]

    return test_images, test_labels


# Show a 28x28 image that is passed to model.predict()
def plot_image(image):
    img = image.reshape(1, 28, 28)
    plt.figure()
    plt.imshow(img[0])
    plt.show()


# Load (a single) image and predict class
def classify_image(image_path, saved_weights):
    """
    :param image_path: Pass the path of the directory you'd like to read (ex: "example_5.png")
    :param saved_weights: Pass the weights of the model you are using for classification

    This function classifies a single image (.png file)

        The image is loaded using the load_image function found in Prediction.py
        The model is loaded using load_model from keras.models

        The probabilities for each digit are calculated using model.predict()
        The predicted class (digit) is returned via the argmax of these probabilities
        The certainty is the probability for the predicted class

        Print the predicted class and certainty level

        OPTIONAL: plot the image using plot_image function found in Prediction.py

    """
    # Load Image
    img = load_image('%s' % image_path)
    # Load model weights
    model = load_model('%s' % saved_weights)

    # Print probabilities
    digit_probs = model.predict(img)
    print(digit_probs)

    # Print digit with highest probability, and certainty level
    digit = np.argmax(digit_probs, axis=-1)
    percent_certain = digit_probs[-1, digit[0]]
    print(digit[0], "with {:.3%}".format(percent_certain), "certainty\n")

    # DEBUG: Show image to check how the .png has translated to the 28x28 array
    # plot_image(img)


# Classify every image in a given directory
def classify_directory(directory_path, saved_weights):
    """
    :param directory_path: Pass the path of the directory you'd like to read (ex: "PredictionImages/Raiford_Handwritten")
    :param saved_weights: Pass the weights of the model you are using for classification

    This function classifies every image (.png files) in a directory.

        The model is loaded using load_model from keras.models
        The function iterates through every non-hidden file in a directory. For each iteration:
            the image is loaded using the load_image function found in Prediction.py
            the probabilities for each digit are calculated using model.predict()
            the predicted class (digit) is returned via the argmax of these probabilities
            the certainty is the probability for the predicted class

            print the name of the file, the predicted class, and certainty level

            OPTIONAL: plot the image using plot_image function found in Prediction.py

    """

    # Load weights
    model = load_model('%s' % saved_weights)

    # For every image x in directory, classify x as 0-9
    file_list = listdir(directory_path)
    file_list.sort()
    for filename in file_list:
        # All non-hidden files (this is a Mac-specific solution, if you are using a Windows machine you may have to
        # use a different workaround to prevent this program from accessing hidden files)
        if not filename.startswith('.'):
            # Show probabilities for each class (if you use this, comment out the rest of this function)
            # run_example("%s/%s" % (directory, filename), saved_weights)

            # Load image
            img = load_image("%s/%s" % (directory_path, filename))

            # Prediction and certainty level
            digit_probs = model.predict(img)
            print(digit_probs)
            digit = np.argmax(digit_probs, axis=-1)
            percent_certain = digit_probs[-1, digit[0]]

            # Print filename, prediction, and certainty level
            print(filename, digit[0], percent_certain, "\n")

            # DEBUG: Show image to check how the .png has translated to the 28x28 array
            # plot_image(img)


# Classify a number of images from MNIST dataset
def classify_mnist(saved_weights, num_selected=100):
    """
    :param saved_weights: Pass the weights of the model you are using for classification
    :param num_selected: How large of a sample to take from the MNIST testing dataset

    NOTE: This testing is already done on the full dataset (num_selected == 10000) using final_test in Testing.py
        This is just a user-friendly way to see how the classification performs for individual images in the dataset

    """

    # Load images and labels from the MNIST testing dataset
    images, labels = load_testing()
    # Check that the num_selected parameter is not greater than the length of the dataset
    if num_selected > len(images):
        # If it is, then set num_selected to the length of the dataset
        num_selected = len(images)

    # Pick num_selected random indices without replacement
    indices = np.random.choice(len(images), num_selected, replace=False)

    # Create samples using those indices
    images_sample = images[indices]
    labels_sample = labels[indices]

    # Load model
    model = load_model('%s' % saved_weights)

    # Keep track of number of correct matches
    num_correct = 0

    # Iterate through the entire sample
    for i in range(num_selected):
        # Define variables for easier calls
        current_image = images_sample[i]
        current_label = np.argmax(labels_sample[i])

        # Print probabilities
        digit_probs = model.predict(current_image)
        print(digit_probs)

        # Print digit with highest probability, certainty level, and correct label
        digit = np.argmax(digit_probs, axis=-1)
        percent_certain = digit_probs[-1, digit[0]]
        print(digit[0], "with {:.3%}".format(percent_certain), "certainty")
        print("\tCorrect label is:", current_label)

        # Print correct or incorrect and update num_correct
        if digit[0] == current_label:
            print("\tCORRECT\n")
            num_correct += 1
        else:
            print("\tINCORRECT\n")

        # Show image
        # plot_image(current_image)

    # Print final success rate statistics
    pct_correct = num_correct / num_selected
    print("Tested on a random sample of", num_selected, "images from MNIST test dataset")
    print("Achieved", "{:.3%}".format(pct_correct), "accuracy overall (%f/%f)" % (num_correct, num_selected))


# Entry Point
# Initialize weights and example images to load
test_image = "MNIST/example_5.png"
test_directory = "MNIST/PredictionImages/Raiford_Handwritten"
weights = "MNIST/final_model.h5"

# Pick the classification you'd like to run
# classify_image(test_image, weights)
classify_directory(test_directory, weights)
# classify_mnist(weights, 1000)

"""
Helpful articles:
https://stackoverflow.com/questions/47435526/what-is-the-meaning-of-axis-1-in-keras-argmax
https://stackoverflow.com/questions/15235823/how-to-ignore-hidden-files-in-python-functions
"""
