#!/usr/bin/env python

import matplotlib.pyplot as plt

def plot_training_history(history, save_plots=False, save_dir="../plots/"):
    """
    Plots training and validation accuracy and loss from a Keras history object.

    Args:
        history: Keras History object returned by model.fit()
        save_plots (bool): If True, saves the plots as PNG files.
        save_dir (str): Directory to save the plots (if save_plots=True).
    """
    # Create save directory if needed
    if save_plots:
        import os
        os.makedirs(save_dir, exist_ok=True)

    # Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], marker='o')
    plt.plot(history.history['val_accuracy'], marker='o')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"])
    plt.grid(True)
    if save_plots:
        plt.savefig(f"{save_dir}/accuracy.png")
    plt.show()

    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'], marker='o')
    plt.plot(history.history['val_loss'], marker='o')
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])
    plt.grid(True)
    if save_plots:
        plt.savefig(f"{save_dir}/loss.png")
    plt.show()

