import os.path

import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(train_accs, test_accs, sensitivities, specificities, confusion_matrices, plotname=None):
    epochs = range(len(train_accs))

    # Plot training and validation accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_accs, label='Training Accuracy', marker='o')
    plt.plot(epochs, test_accs, label='Validation Accuracy', marker='o')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend next to the plot

    # Plot sensitivity
    plt.subplot(2, 2, 2)
    for i in range(len(sensitivities[0])):
        plt.plot(epochs, [sensitivity[i] for sensitivity in sensitivities], label=f'Class {i}', marker='o')
    plt.title('Sensitivity over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Sensitivity')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend next to the plot
    plt.ylim(0, 1.1)

    # Plot specificity
    plt.subplot(2, 2, 3)
    for i in range(len(specificities[0])):
        plt.plot(epochs, [specificity[i] for specificity in specificities], label=f'Class {i}', marker='o')
    plt.title('Specificity over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Specificity')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend next to the plot
    plt.ylim(0, 1.1)

    # Plot confusion matrix for the last epoch
    plt.subplot(2, 2, 4)
    cm = confusion_matrices[-1]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Adjust layout to make space for legends

    if plotname is not None:
        os.makedirs(os.path.dirname(plotname), exist_ok=True)
        plt.savefig(plotname, dpi=300)
    plt.show()
