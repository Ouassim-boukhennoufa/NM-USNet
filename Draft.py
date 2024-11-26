import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from unet_model import build_unet  # Import the U-Net model from another file

# Initialize alpha and beta as trainable variables
alpha = tf.Variable(0.5, trainable=True, dtype=tf.float32)
beta = tf.Variable(0.5, trainable=True, dtype=tf.float32)

# Normalize alpha and beta to ensure alpha + beta = 1
def normalize_alpha_beta(alpha, beta):
    total = alpha + beta
    return alpha / total, beta / total

# Define custom loss
def combined_loss(y_true, y_pred, additional_loss_fn):
    # Mean Squared Error for U-Net
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    # Additional loss (e.g., similarity or other metric)
    additional_loss = additional_loss_fn(y_true, y_pred)
    # Normalize alpha and beta
    norm_alpha, norm_beta = normalize_alpha_beta(alpha, beta)
    # Combined global loss
    return norm_alpha * mse_loss + norm_beta * additional_loss

# Training step with dynamic alpha and beta adjustment
@tf.function
def train_step(model, x_batch, y_batch, optimizer, additional_loss_fn):
    with tf.GradientTape() as tape:
        # Forward pass through the model
        y_pred = model(x_batch, training=True)
        # Compute combined loss
        loss = combined_loss(y_batch, y_pred, additional_loss_fn)
    
    # Compute gradients for model weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Adjust alpha and beta dynamically
    with tf.GradientTape() as tape_alpha_beta:
        loss_alpha_beta = combined_loss(y_batch, y_pred, additional_loss_fn)
    alpha_beta_grads = tape_alpha_beta.gradient(loss_alpha_beta, [alpha, beta])
    alpha_beta_optimizer.apply_gradients(zip(alpha_beta_grads, [alpha, beta]))
    
    return loss

# 10-Fold cross validation
def train_unet_with_cv(X, y, additional_loss_fn, num_folds=10, epochs=500, batch_size=32):
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
                train_loss = train_step(model, x_batch, y_batch, optimizer, additional_loss_fn)

            # Validation step
            y_pred_val = model(X_val, training=False)
            val_loss = combined_loss(y_val, y_pred_val, additional_loss_fn)
            print(f"Fold {fold_no}, Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.numpy()}, Val Loss: {val_loss.numpy()}, Alpha: {alpha.numpy()}, Beta: {beta.numpy()}")
        
        # Store validation results for this fold
        results.append(val_loss.numpy())
        fold_no += 1

    print(f"\nCross-validation results: {results}")
    print(f"Mean Validation Loss: {np.mean(results)}")

# Example usage
if __name__ == "__main__":
    # Placeholder for preprocessed data (replace with actual data)
    X = np.random.rand(1000, 128, 128, 1)  # Example input data
    y = np.random.rand(1000, 128, 128, 1)  # Example target data

    # Example additional loss function (e.g., mean absolute error)
    def additional_loss_fn(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    # Train U-Net with 10-Fold Cross Validation
    train_unet_with_cv(X, y, additional_loss_fn, num_folds=10, epochs=500, batch_size=32)
