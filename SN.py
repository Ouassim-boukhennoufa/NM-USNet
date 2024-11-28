import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout
import numpy as np


class SiameseNetwork:
    """
    Siamese Network for learning similarity between pairs of images.

    This network takes in pairs of images (real and random noise) and learns 
    to output a similarity score based on their embeddings.

    Attributes:
        input_shape (tuple): Shape of the input images, default is (128, 128, 1).
        base_model (tf.keras.Model): The base embedding model for feature extraction.
        siamese (tf.keras.Model): The Siamese network model.
    """

    def __init__(self, input_shape=(128, 128, 1)):
        """
        Initializes the Siamese Network.

        Args:
            input_shape (tuple): Shape of the input images (height, width, channels).
        """
        self.input_shape = input_shape
        self.base_model = self._build_base_model()
        self.siamese = self._build_siamese_network()

    def _build_base_model(self):
        """
        Builds the base model used for embedding feature extraction.

        Returns:
            tf.keras.Model: A Keras model for embedding generation.
        """
        input_tensor = Input(shape=self.input_shape)

        x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64, (7, 7), activation='relu')(x)
        x = Dropout(0.2)(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)
        x = Dense(4096, activation='sigmoid')(x)

        model = Model(input_tensor, x)
        return model

    def _build_siamese_network(self):
        """
        Builds the full Siamese network that computes similarity scores between image pairs.

        Returns:
            tf.keras.Model: A Keras model representing the Siamese network.
        """
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)

        embedding_a = self.base_model(input_a)
        embedding_b = self.base_model(input_b)

        # Compute the distance between embeddings
        distance = Lambda(self._euclidean_distance)([embedding_a, embedding_b])

        model = Model(inputs=[input_a, input_b], outputs=distance)
        return model

    @staticmethod
    def _euclidean_distance(vectors):
        """
        Computes the Euclidean distance between two vectors.

        Args:
            vectors (list): A list containing two tensors (a and b).

        Returns:
            tf.Tensor: A tensor representing the Euclidean distance.
        """
        a, b = vectors
        return tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=1, keepdims=True))

    def compile(self, optimizer='adam', loss='mse'):
        """
        Compiles the Siamese network with the given optimizer and loss function.

        Args:
            optimizer (str or tf.keras.optimizers.Optimizer): The optimizer to use.
            loss (str or tf.keras.losses.Loss): The loss function to use.
        """
        self.siamese.compile(optimizer=optimizer, loss=loss)

    def train(self, x_train, y_train, epochs=1000, batch_size=20):
        """
        Trains the Siamese network on the given data.

        Args:
            x_train (numpy.ndarray): Training data, shape (num_samples, 2, height, width, channels).
                                     Contains pairs of images.
            y_train (numpy.ndarray): Labels for the training data.
            epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
        """
        self.siamese.fit(
            [x_train[:, 0], x_train[:, 1]], 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size
        )


"""
How to use the Network:

siamese_net = SiameseNetwork(input_shape=(128, 128, 1))


siamese_net.compile(optimizer='adam', loss='mse')


# x_train shape: (num_samples, 2, 128, 128, 1)
# y_train shape: (num_samples,)
siamese_net.train(x_train, y_train, epochs=1000, batch_size=20)

"""
