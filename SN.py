#SN.py
#Contains the siamese network

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout
import numpy as np

class SiameseNetwork:
    def __init__(self, input_shape=(128, 128, 1)):
        self.input_shape = input_shape
        self.base_model = self.base_model()
        self.siamese = self.siamese_network()

    def base_model(self):
        input = Input(shape=self.input_shape)
        
        x = Conv2D(32, (3,3), activation='relu')(input)
        x = MaxPooling2D((2,2))(x)
        
        x = Conv2D(64, (7,7), activation='relu')(x)
        x = Dropout(0.2)(x)
        x = MaxPooling2D((2,2))(x)
        
        x = Flatten()(x)
        
        x = Dense(4096, activation='sigmoid')(x)
        
        model = Model(input, x)
        
        return model
    
    def siamese_network(self):
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)
        
        embedding_a = self.base_model(input_a)
        embedding_b = self.base_model(input_b)
        
        distance = Lambda(self.euclidean_distance)([embedding_a, embedding_b])
        
        model = Model(inputs=[input_a, input_b], outputs=distance)
        return model
    
    @staticmethod
    def euclidean_distance(vectors):
        a, b = vectors
        return tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=1, keepdims=True))
    
    def compile(self, optimizer='adam', loss='mse'):
        self.siamese.compile(optimizer=optimizer, loss=loss)
    
    def train(self, x_train, y_train, epochs=1000, batch_size=20):
        self.siamese.fit([x_train[:, 0], x_train[:, 1]], y_train, epochs=epochs, batch_size=batch_size)
    
    


