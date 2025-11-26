#!/usr/bin/env python

from train import get_training_history 
import matplotlib.pyplot as plt
import os

history = get_training_history()
def plot_training_history(history, save_plots=False, save_dir="../plots/"):
    """
    Plots training and validation accuracy and loss from a Keras history object.

    Args:
        history: Keras History object returned by model.fit()
        save_plots (bool): If True, saves the plots as PNG files.
        save_dir (str): Directory to save the plots (if save_plots=True).
    """
    # Make path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.abspath(os.path.join(script_dir, save_dir))

    # Create directory if saving
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

    # Plot accuracy
    plt.figure()

    plt.plot(history['accuracy'], marker='o')
    plt.plot(history['val_accuracy'], marker='o')
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

    plt.plot(history['loss'], marker='o')
    plt.plot(history['val_loss'], marker='o')
    
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])
    plt.grid(True)
    
    if save_plots:
        plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.close()

    print(f"Plots saved to: {save_dir}" if save_plots else "Plots displayed.")

if __name__ == "__main__":
    plot_training_history(history, save_plots=True)
