from keras.layers import *

class Localization(Layer):
    def __init__(self):
        super(Localization, self).__init__()
        self.pool1 = MaxPool2D()
        self.conv1 = Conv2D(20, [5, 5], activation='relu')
        self.pool2 = MaxPool2D()
        self.conv2 = Conv2D(20, [5, 5], activation='relu')
        self.flatten = Flatten()
        self.fc1 = Dense(20, activation='relu')
        self.fc2 = Dense(6, activation=None, bias_initializer=tf.keras.initializers.constant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), kernel_initializer='zeros')

    def build(self, input_shape):
        print("Building Localization Network with input shape:", input_shape)

    def compute_output_shape(self, input_shape):
        return [None, 6]

    def call(self, inputs):
        X = self.conv1(inputs)
        X = self.pool1(X)
        X = self.conv2(X)
        X = self.pool2(X)
        X = self.flatten(X)
        X = self.fc1(X)
        theta = self.fc2(X)
        theta = tf.keras.layers.Reshape((2, 3))(theta)
        return theta