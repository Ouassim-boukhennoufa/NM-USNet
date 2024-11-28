import tensorflow as tf
from tensorflow.keras import layers, models, backend as K


def combined_loss(y_true, y_pred, alpha, beta, siamese_model):
    """
    Computes the combined loss using UNet loss and Siamese network similarity loss.

    Args:
        y_true (tf.Tensor): Ground truth tensor.
        y_pred (tf.Tensor): Predicted tensor.
        alpha (float): Weight for the reconstruction loss (UNet loss).
        beta (float): Weight for the similarity loss (Siamese network loss).
        siamese_model (tf.keras.Model): The Siamese network model to compute similarity.

    Returns:
        tf.Tensor: Total combined loss value.
    """
    # Reconstruction loss (UNet loss)
    unet_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)

    # Similarity score from Siamese Network
    similarity_score = siamese_model([y_true, y_pred])
    siamese_loss = 1 - tf.reduce_mean(similarity_score)

    # Total combined loss
    total_loss = alpha * unet_loss + beta * siamese_loss
    return total_loss


def pearson_correlation_metric(y_true, y_pred):
    """
    Computes the Pearson correlation coefficient between the true and predicted values.

    Args:
        y_true (tf.Tensor): Ground truth tensor.
        y_pred (tf.Tensor): Predicted tensor.

    Returns:
        tf.Tensor: Pearson correlation coefficient.
    """
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


# Enable Eager execution (needed for Pearson correlation metric)
tf.compat.v1.enable_eager_execution()


def unet(input_shape):
    """
    Constructs a UNet generator model for image-to-image translation.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).

    Returns:
        tf.keras.Model: UNet model.
    """
    # Define two input layers for the pair of images
    input_layer1 = layers.Input(shape=input_shape, name='input1')
    input_layer2 = layers.Input(shape=input_shape, name='input2')

    # Encoder part
    x = layers.Concatenate()([input_layer1, input_layer2])
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # Additional encoder layers can be added here if needed.

    # Decoder part
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    # Additional decoder layers can be added here if needed.

    # Output layer
    output_layer = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='generated_image')(x)

    # Define the model
    generator_model = models.Model(inputs=[input_layer1, input_layer2], outputs=output_layer, name='generator')
    return generator_model


# Define the input shape for images
input_shape = (128, 128, 1)

# Build the UNet generator
generator = unet(input_shape)

# Print the model summary
generator.summary()


#Fit function (training)
#Adjust 'input1', 'input2', and 'generated_image' as your data
generator.compile(optimizer='adam', loss=combined_loss, metrics=[pearson_correlation_metric])
history = generator.fit(
    {'input_image1': train_img1, 'input_image2': train_img2},
    {'generated_image': train_output},
    validation_split = 0.1,
    epochs=500)



#Test and adjust with your test data
test_loss, test_metric = generator.evaluate([test_img1, test_img2] ,test_output)
print(f'Test Loss: {test_loss}, Test_Metric: {test_metric}')
