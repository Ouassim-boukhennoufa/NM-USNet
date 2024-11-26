from keras.models import load_model
import tensorflow as tf
import numpy as np


#Customize the path according to the saved SN
path = "SN.h5"
model = load_model(path, custom_objects={'ContrastiveLoss': ContrastiveLoss})

def compute_similarity(image1, image2):
    y_true = tf.expand_dims(y_true, axis=0) if len(y_true.shape) == 3 else y_true  # Add batch dimension if necessary
    y_pred = tf.expand_dims(y_pred, axis=0) if len(y_pred.shape) == 3 else y_pred  # Add batch dimension if necessary
    
    # Get the similarity score (Euclidean distance) using the Siamese network
    similarity_score = model.predict([y_true, y_pred])
    return similarity_score
