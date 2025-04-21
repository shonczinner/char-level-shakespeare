import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from utils.constants import SAVE_PATH

def compare_experiments(model_types):
    # Generate color map
    colors = cm.get_cmap('tab10', len(model_types))
    model_colors = {model: colors(i) for i, model in enumerate(model_types)}

    # Initialize figures and axes
    fig_loss_epoch, ax_loss_epoch = plt.subplots()
    fig_acc_epoch, ax_acc_epoch = plt.subplots()
    fig_loss_compute, ax_loss_compute = plt.subplots()
    fig_acc_compute, ax_acc_compute = plt.subplots()

    for model_type in model_types:
        save_path = os.path.join(SAVE_PATH, model_type)
        metrics_path = os.path.join(save_path, "metrics.csv")

        if not os.path.exists(metrics_path):
            print(f"Metrics file not found for {model_type}, skipping.")
            continue

        metrics = pd.read_csv(metrics_path, header=0)
        train_losses = metrics["train_losses"].tolist()
        val_losses = metrics["val_losses"].tolist()
        train_accs = metrics["train_accs"].tolist()
        val_accs = metrics["val_accs"].tolist()
        compute = metrics["compute"].tolist()
        epochs = list(range(1, len(train_losses) + 1))
        color = model_colors[model_type]

        # Loss vs Epoch
        ax_loss_epoch.plot(epochs, train_losses, label=f'{model_type} - Train', linestyle='-', color=color)
        ax_loss_epoch.plot(epochs, val_losses, label=f'{model_type} - Val', linestyle='--', color=color)

        # Accuracy vs Epoch
        ax_acc_epoch.plot(epochs, train_accs, label=f'{model_type} - Train', linestyle='-', color=color)
        ax_acc_epoch.plot(epochs, val_accs, label=f'{model_type} - Val', linestyle='--', color=color)

        # Loss vs Compute
        ax_loss_compute.plot(compute, train_losses, label=f'{model_type} - Train', linestyle='-', color=color)
        ax_loss_compute.plot(compute, val_losses, label=f'{model_type} - Val', linestyle='--', color=color)

        # Accuracy vs Compute
        ax_acc_compute.plot(compute, train_accs, label=f'{model_type} - Train', linestyle='-', color=color)
        ax_acc_compute.plot(compute, val_accs, label=f'{model_type} - Val', linestyle='--', color=color)

    # Finalize and save each plot
    for ax, title, xlabel, ylabel, filename in [
        (ax_loss_epoch, "Loss vs Epoch", "Epoch", "Loss", "loss_vs_epoch.png"),
        (ax_acc_epoch, "Accuracy vs Epoch", "Epoch", "Accuracy", "acc_vs_epoch.png"),
        (ax_loss_compute, "Loss vs Compute", "Train Compute In Seconds", "Loss", "loss_vs_compute.png"),
        (ax_acc_compute, "Accuracy vs Compute", "Train Compute In Seconds", "Accuracy", "acc_vs_compute.png")
    ]:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.figure.tight_layout()
        ax.figure.savefig(filename)
        plt.close(ax.figure)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare experiment metrics across models.")
    parser.add_argument(
        "--model_types",
        nargs="+",
        default=["rnn", "cnn", "transformer"],
        help="List of model types to compare (default: rnn cnn transformer)"
    )
    args = parser.parse_args()

    compare_experiments(args.model_types)
