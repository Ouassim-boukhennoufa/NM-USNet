#Various functions if needed

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
cmap = plt.cm.gray_r

#Display two images if needed
def DisplayTwo(image1, image2, message1, message2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image1, cmap = cmap)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    axes[0].set_title(message1)
    cbar1 = fig.colorbar(axes[0].imshow(image1, cmap=cmap), ax=axes[0])
    cbar1.ax.tick_params(labelsize=20)
    #axes[0].axis('off')
    axes[1].imshow(image2, cmap=cmap)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[1].set_title(message2)
    cbar2 = fig.colorbar(axes[1].imshow(image2, cmap=cmap), ax=axes[1])
    cbar2.ax.tick_params(labelsize=20)
    #axes[1].axis('off')
    fig.tight_layout()
    plt.show()

# Plot training & validation pearson correlation values    
def PlotTestTrain(history):
    
    plt.plot(history.history['pearson_correlation_metric'])
    plt.plot(history.history['val_pearson_correlation_metric'])
    plt.title('Model Pearson correlatn')
    plt.ylabel('Correlation')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
