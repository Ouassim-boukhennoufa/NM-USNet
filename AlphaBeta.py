import tensorflow as tf

# Define alpha and beta as trainable variables
alpha = tf.Variable(0.5, trainable=True, dtype=tf.float32)
beta = tf.Variable(0.5, trainable=True, dtype=tf.float32)

# Ensure alpha + beta = 1 by normalization function
def normalize_alpha_beta(alpha, beta):
    total = alpha + beta
    return alpha / total, beta / total

def train_with_cross_validation(X, y, num_folds=10, epochs=500):
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_no = 1

    for train_idx, val_idx in kfold.split(X, y):
        print(f"\nTraining on fold {fold_no}...")
        
        # Initialize a new model for each fold
        model = build_unet_model()
        optimizer = tf.optimizers.Adam(learning_rate=0.001)

        # Split the data into training and validation sets for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        for epoch in range(epochs):
            # Iterate through each batch of training data
            for batch in range(len(X_train)):
                x_batch = X_train[batch]
                y_batch = y_train[batch]
                
                # Perform training step
                loss = train_step(model, x_batch, y_batch)

            # Validation at the end of each epoch
            y_pred_val = model(X_val, training=False)
            val_loss = custom_loss(y_val, y_pred_val)
            print(f"Fold {fold_no}, Epoch {epoch+1}/{epochs}, Train Loss: {loss.numpy()}, Val Loss: {val_loss.numpy()}, Alpha: {alpha.numpy()}, Beta: {beta.numpy()}")
        
        fold_no += 1

# Define the custom loss function L = alpha*L1 + beta*L2
def custom_loss(y_true, y_pred):
    L1 = compute_L1(y_true, y_pred)
    L2 = compute_L2(y_true, y_pred)
    norm_alpha, norm_beta = normalize_alpha_beta(alpha, beta)
    return norm_alpha * L1 + norm_beta * L2

# Optimizer for alpha and beta adjustment
alpha_beta_optimizer = tf.optimizers.Adam(learning_rate=0.01)

# Training step function with dynamic alpha and beta adjustment
@tf.function
def train_step(model, x, y_true):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = custom_loss(y_true, y_pred)
    
    # Calculate gradients of the model weights
    gradients = tape.gradient(loss, model.trainable_variables)
    # Update model weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Adjust alpha and beta based on the gradients of the loss w.r.t. alpha and beta
    with tf.GradientTape() as tape_alpha_beta:
        loss_alpha_beta = custom_loss(y_true, model(x, training=True))
    alpha_beta_grads = tape_alpha_beta.gradient(loss_alpha_beta, [alpha, beta])
    alpha_beta_optimizer.apply_gradients(zip(alpha_beta_grads, [alpha, beta]))

    return loss
