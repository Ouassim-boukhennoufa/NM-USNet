import tensorflow as tf
from tensorflow.keras import layers, models
import os
from tensorflow.keras.losses import Loss
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
cmap = plt.cm.gray_r
from tensorflow.keras import backend as K
from siamese_network import SiameseNetwork

# Freeze the Siamese network weights (since it's pretrained)
siamese_model.trainable = False

# Define Custom Loss Function
def combined_loss(y_true, y_pred, alpha, beta):
    # Reconstruction loss
    unet_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    # Similarity score from Siamese Network
    S = siamese_model([y_true, y_pred]) 
    
    siamese_loss = (1-tf.reduce_mean(S))
    
    total_loss = alpha * unet_loss + beta * siamese_loss
    return total_loss


def pearson_correlation_metric(y_true, y_pred):
    # Flatten the tensors
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    # Compute the means
    mean_true = K.mean(y_true_flat)
    mean_pred = K.mean(y_pred_flat)

    # Compute covariance and variances
    covariance = K.mean((y_true_flat - mean_true) * (y_pred_flat - mean_pred))
    variance_true = K.mean(K.square(y_true_flat - mean_true))
    variance_pred = K.mean(K.square(y_pred_flat - mean_pred))

    # Compute Pearson correlation
    pearson_corr = covariance / (K.sqrt(variance_true) * K.sqrt(variance_pred) + K.epsilon())

    return pearson_corr

#For Eager execution, necessary for the pearson metric    
tf.compat.v1.enable_eager_execution()

#UNET architecture
def Unet(input_shape):
    input_layer1 = layers.Input(shape=input_shape, name='input1')
    input_layer2 = layers.Input(shape=input_shape, name='input2')

    #Encoder part
    x = layers.Concatenate()([input_layer1, input_layer2])
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # Add more encoder layers as needed

    #Decoder part
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    # Add more decoder layers as needed

    #Output layer
    output_layer = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='generated_image')(x)

    # Model
    generator_model = models.Model(inputs=[input_layer1, input_layer2], outputs=output_layer, name='generator')
    return generator_model

#Adjust based on your image size and channels    
input_shape = (128, 128, 1)  
generator = Unet(input_shape)
generator.summary()

#Fit function (training)
#Adjust 'input1', 'input2', and 'generated_image' as your data
generator.compile(optimizer='adam', loss='mean_squared_error', metrics=[pearson_correlation_metric])
history = generator.fit(
    {'input_image1': train_img1, 'input_image2': train_img2},
    {'generated_image': train_output},
    validation_split = 0.1,
    epochs=500)


# Plot training & validation pearson correlation values
plt.plot(history.history['pearson_correlation_metric'])
plt.plot(history.history['val_pearson_correlation_metric'])
plt.title('Model Pearson correlatn')
plt.ylabel('Correlation')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#Test and adjust with your test data
test_loss, test_metric = generator.evaluate([test_img1, test_img2] ,test_output)
print(f'Test Loss: {test_loss}, Test_Metric: {test_metric}')
