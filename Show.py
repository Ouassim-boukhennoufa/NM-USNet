#Various functions if needed to visualize 

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define colormap for consistent gray scale visualization (most adapted for this use case)
cmap = plt.cm.gray_r


def display_two(image1, image2, message1, message2):
    """
    Displays two images side by side with titles and colorbars.

    Args:
        image1 (numpy.ndarray or PIL.Image): The first image to display.
        image2 (numpy.ndarray or PIL.Image): The second image to display.
        message1 (str): Title for the first image.
        message2 (str): Title for the second image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the first image
    im1 = axes[0].imshow(image1, cmap=cmap)
    axes[0].set_title(message1)
    cbar1 = fig.colorbar(im1, ax=axes[0])
    cbar1.ax.tick_params(labelsize=10)

    # Display the second image
    im2 = axes[1].imshow(image2, cmap=cmap)
    axes[1].set_title(message2)
    cbar2 = fig.colorbar(im2, ax=axes[1])
    cbar2.ax.tick_params(labelsize=10)

    # Adjust layout and show the figure
    fig.tight_layout()
    plt.show()


def plot_test_train(history):
    """
    Plots the training and validation Pearson correlation metrics over epochs.

    Args:
        history (keras.callbacks.History): The history object containing training metrics.
                                           Must include 'pearson_correlation_metric' 
                                           and 'val_pearson_correlation_metric'.
    """
    plt.plot(history.history['pearson_correlation_metric'], label='Train')
    plt.plot(history.history['val_pearson_correlation_metric'], label='Validation')
    plt.title('Model Pearson Correlation')
    plt.ylabel('Correlation')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

