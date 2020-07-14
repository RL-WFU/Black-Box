"""
A pure TensorFlow implementation of a convolutional neural network.
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools

import tensorflow as tf
import tensorflow.keras as keras

from cleverhans import initializers
from cleverhans.model import Model
from WeightUploading.LoadWeights import *


class ModelBasicCNN(Model):
  def __init__(self, scope, nb_classes, nb_filters, weights, **kwargs):
    del kwargs
    Model.__init__(self, scope, nb_classes, locals())
    self.nb_filters = 32

    self.weights = weights

    self.layer_info = np_weights(self.weights)
    print(self.layer_info[0][0].shape, self.layer_info[0][1].shape)
    print(self.layer_info[1][0].shape, self.layer_info[1][1].shape)
    print(self.layer_info[2][0].shape, self.layer_info[2][1].shape)
    # Do a dummy run of fprop to make sure the variables are created from
    # the start
    self.fprop(tf.placeholder(tf.float32, [1, 28, 28, 1]))
    # Put a reference to the params in self so that the params get pickled
    self.params = self.get_params()

  def fprop(self, x, **kwargs):
    del kwargs
    my_conv = functools.partial(
        tf.layers.conv2d, activation=tf.nn.relu,
        kernel_initializer=initializers.HeReLuNormalInitializer)

    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      y = tf.layers.conv2d(x, self.nb_filters, 3, 1, padding='same', activation=tf.nn.relu,
                           kernel_initializer=initialize_weights(self.layer_info[0][0]),
                           bias_initializer=initialize_weights(self.layer_info[0][1]))
      #y = my_conv(x, self.nb_filters, 3, strides=1, padding='same')
      y = tf.layers.MaxPooling2D((2,2), 2)(y)
      #y = my_conv(y, 2 * self.nb_filters, 3, strides=1, padding='valid')
      #y = my_conv(y, 2 * self.nb_filters, 3, strides=1, padding='valid')
      #y = tf.layers.MaxPooling2D((2,2), 1)(y)
      y = tf.layers.flatten(y)

      dense = tf.layers.dense(y, 100, activation='relu',
                              kernel_initializer=initialize_weights(self.layer_info[1][0]),
                              bias_initializer=initialize_weights(self.layer_info[1][1]))
      logits = tf.layers.dense(
          dense, self.nb_classes,
          kernel_initializer=initialize_weights(self.layer_info[2][0]),
          bias_initializer=initialize_weights(self.layer_info[2][1]))
      return {self.O_LOGITS: logits,
              self.O_PROBS: tf.nn.softmax(logits=logits)}


def build_baseline_with_weights(layer_info):
    """
    :param layer_info: a list of weights, with at least two axes. The first axis specifies the layer index, not
    including layers with no weights, like pooling or flatten layers. The second axis denotes either weights or biases,
    index 0 being weights, and index 1 being biases. So layer_info[0][0] denotes the weights of the first layer
    If weights are not in the above format, run them through the np_weights function

    :return: model: a keras model object which holds the model layers and parameters. Calling .predict_on_batch on this
    model object will return an array of shape (1, 10), containing the probabilities of each label
    """
    model = Sequential()

    # CONSTRUCT LAYERS

    #kernel_initializer and bias_initializer are replaced with Initializer subclassing (talked about above).
    #Pass as argument to initialize_weights the corresponding weights or biases of the layer. In this case, it is
    #the first layer's weights and biases, so layer_info[0][0] and layer_info[0][1] are passed, respectively
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=initialize_weights(layer_info[0][0]), bias_initializer=initialize_weights(layer_info[0][1]), input_shape=(28, 28, 1)))
    # Max pooling layer with 2x2 filter
    #Max pooling and flatten do not have weights
    model.add(MaxPooling2D((2, 2)))
    # Flatten filter maps to pass to classifier
    model.add(Flatten())
    # Fully-connected layer with 100 nodes, ReLU activation function
    #Second layer that needs weights from layer_info
    model.add(Dense(100, activation='relu', kernel_initializer=initialize_weights(layer_info[1][0]), bias_initializer=initialize_weights(layer_info[1][1])))
    # Fully-connected output layer with 10 nodes (for the labels [0,9])
    #Third, and last layer with initialized weights
    model.add(Dense(10, activation='softmax', kernel_initializer=initialize_weights(layer_info[2][0]), bias_initializer=initialize_weights(layer_info[2][1])))

    return model

class MyModel(tf.keras.Model):
    def __init__(self):

        super(MyModel, self).__init__()
        #self.weight_info = weights
        self.conv = keras.layers.Conv2D(32, (3,3), input_shape=(28, 28, 1), activation='relu')
        self.pool = keras.layers.MaxPooling2D((2, 2))
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(100, activation='relu')
        self.logits = keras.layers.Dense(10, activation=None)

    def call(self, inputs):
        return self.get_logits(inputs)

    def get_logits(self, x):
        y = self.conv(x)
        y = self.pool(y)
        y = self.flatten(y)
        y = self.dense(y)
        logits = self.logits(y)
        return logits


"""
with tf.Session() as sess:
    model = MyModel()
    sess.run(tf.global_variables_initializer())
    model.build((1, 28, 28, 1))
    model.load_weights("../../MNIST/final_model.h5")
    model.summary()
    img = load_image("../../MNIST/PredictionImages/example_5.png")
    results = tf.nn.softmax(model.predict_on_batch(img)).eval()
    print(results)
"""
