import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from unet_model import build_unet  # Import the U-Net model from another file

# Initialize alpha and beta as trainable variables
alpha = tf.Variable(0.5, trainable=True, dtype=tf.float32)
beta = tf.Variable(0.5, trainable=True, dtype=tf.float32)


def normalize_alpha_beta(alpha, beta):
    """
    Normalizes alpha and beta to ensure alpha + beta = 1.

    Args:
        alpha (tf.Variable): Trainable weight for mean squared error loss.
        beta (tf.Variable): Trainable weight for additional loss.

    Returns:
        tuple: Normalized alpha and beta values.
    """
    total = alpha + beta
    return alpha / total, beta / total


def combined_loss(y_true, y_pred, siamese_network_loss):
    """
    Computes the combined loss using mean squared error and an additional loss term.

    Args:
        y_true (tf.Tensor): Ground truth labels.
        y_pred (tf.Tensor): Predicted outputs.
        siamese_network_loss (callable): Function that computes additional loss.

    Returns:
        tf.Tensor: Combined loss value.
    """
    # Mean Squared Error for U-Net
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # Additional loss (e.g., similarity or other metric)
    additional_loss = siamese_network_loss(y_true, y_pred)

    # Normalize alpha and beta
    norm_alpha, norm_beta = normalize_alpha_beta(alpha, beta)

    # Combined global loss
    return norm_alpha * mse_loss + norm_beta * additional_loss


@tf.function
def train_step(model, x_batch, y_batch, optimizer, siamese_network_loss):
    """
    Performs a single training step for the U-Net model and dynamically adjusts alpha and beta.

    Args:
        model (tf.keras.Model): The U-Net model to be trained.
        x_batch (tf.Tensor): Batch of input images.
        y_batch (tf.Tensor): Batch of ground truth images.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer for training.
        siamese_network_loss (callable): Function that computes additional loss.

    Returns:
        tf.Tensor: Training loss value.
    """
    with tf.GradientTape() as tape:
        # Forward pass through the model
        y_pred = model(x_batch, training=True)
        # Compute combined loss
        loss = combined_loss(y_batch, y_pred, siamese_network_loss)

    # Compute gradients for model weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Adjust alpha and beta dynamically
    with tf.GradientTape() as tape_alpha_beta:
        loss_alpha_beta = combined_loss(y_batch, y_pred, siamese_network_loss)
    alpha_beta_grads = tape_alpha_beta.gradient(loss_alpha_beta, [alpha, beta])
    alpha_beta_optimizer.apply_gradients(zip(alpha_beta_grads, [alpha, beta]))

    return loss


def train_unet_with_cv(X, y, siamese_network_loss, num_folds=10, epochs=500, batch_size=32):
    """
    Trains a U-Net model using 10-fold cross-validation.

    Args:
        X (np.ndarray): Input image data of shape (samples, height, width, channels).
        y (np.ndarray): Ground truth image data of shape (samples, height, width, channels).
        siamese_network_loss (callable): Function that computes additional loss.
        num_folds (int, optional): Number of folds for cross-validation. Defaults to 10.
        epochs (int, optional): Number of epochs for training each fold. Defaults to 500.
        batch_size (int, optional): Batch size for training. Defaults to 32.

    Returns:
        None
    """
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_no = 1
    results = []

    for train_idx, val_idx in kfold.split(X, y):
        print(f"\nTraining on fold {fold_no}...")

        # Create a new U-Net model for this fold
        model = build_unet()
        optimizer = tf.optimizers.Adam(learning_rate=0.001)
        alpha_beta_optimizer = tf.optimizers.Adam(learning_rate=0.01)

        # Split data into training and validation sets
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        for epoch in range(epochs):
            # Shuffle and batch training data
            dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)

            # Train on batches
            for x_batch, y_batch in dataset:
                train_loss = train_step(model, x_batch, y_batch, optimizer, siamese_network_loss)

            # Validation step
            y_pred_val = model(X_val, training=False)
            val_loss = combined_loss(y_val, y_pred_val, siamese_network_loss)
            print(f"Fold {fold_no}, Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss.numpy()}, "
                  f"Val Loss: {val_loss.numpy()}, Alpha: {alpha.numpy()}, Beta: {beta.numpy()}")

        # Store validation results for this fold
        results.append(val_loss.numpy())
        fold_no += 1

    print(f"\nCross-validation results: {results}")
    print(f"Mean Validation Loss: {np.mean(results)}")


if __name__ == "__main__":
    # Placeholder for preprocessed data (replace with actual data), needs to be resized to the number of instances
    X = np.random.rand(1000, 128, 128, 1)  # Input images, here for example they are combined and can be divided in the middle
    y = np.random.rand(1000, 128, 128, 1)  # Ground truth images, same as for inputs

    def siamese_network_loss(y_true, y_pred):
        """
        Computes the Siamese network loss as 1 - similarity.

        Args:
            y_true (tf.Tensor): Ground truth tensor (1 for similar, 0 for dissimilar).
            y_pred (tf.Tensor): Predicted similarity scores.

        Returns:
            tf.Tensor: Siamese network loss value.
        """
        return tf.reduce_mean(tf.abs(y_true - (1 - y_pred)))

    # Train U-Net with 10-Fold Cross Validation
    train_unet_with_cv(X, y, siamese_network_loss, num_folds=10, epochs=500, batch_size=32)

