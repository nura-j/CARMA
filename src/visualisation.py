
import matplotlib.pyplot as plt
import os
from datetime import datetime


def history_plot(history, model_name, task_name):
    """Plot training metrics."""
    # Check for missing keys in history
    required_keys = [
        'train_task_loss', 'train_stability_loss', 'train_mi_loss',
        'val_task_loss', 'val_stability_loss', 'val_mi_loss'
    ]
    for key in required_keys:
        if key not in history:
            raise ValueError(f"Key '{key}' is missing in the history dictionary.")

    plt.figure(figsize=(12, 4))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_task_loss'], label='T Task Loss')
    plt.plot(history['train_stability_loss'], label='T Stability Loss')
    plt.plot(history['train_mi_loss'], label='T MI Loss')
    plt.plot(history['val_task_loss'], label='V Task Loss')
    plt.plot(history['val_stability_loss'], label='V Stability Loss')
    plt.plot(history['val_mi_loss'], label='V MI Loss')
    plt.title('Losses')
    plt.legend()

    # Check if 'train_loss' and 'val_loss' exist for total loss plot
    if 'train_loss' in history and 'val_loss' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Total Loss')
        plt.legend()

    plt.tight_layout()

    # Ensure save directory exists
    parent_dir = os.path.dirname(os.getcwd())
    save_dir = os.path.join(parent_dir, 'plots', model_name, task_name)
    os.makedirs(save_dir, exist_ok=True)

    # Generate save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"{model_name}_carma_{timestamp}_results.pdf")

    # Save and display the plot
    plt.savefig(save_path)
    plt.show()
