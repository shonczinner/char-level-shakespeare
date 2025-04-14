import matplotlib.pyplot as plt
import os

def plot_metrics(train_metrics, val_metrics, save_dir, metric_name):
    """
    Plot and save the training and validation metrics.

    Args:
        train_metrics (list): List of training metric values.
        val_metrics (list): List of validation metric values.
        num_epochs (int): Number of epochs.
        save_dir (str): Directory to save the plots.
        metric_name (str): Name of the metric (e.g., 'Loss', 'Accuracy').
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(train_metrics) + 1), train_metrics, label=f"Train {metric_name}")
    plt.plot(range(1, len(train_metrics) + 1), val_metrics, label=f"Val {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{metric_name.lower()}_curve.png"))
    plt.close()
