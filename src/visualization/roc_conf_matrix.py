import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import sys

sys.path.append("../")

import config.config as config


def load_fold_results(fold_dir):
    """
    Loads saved numpy arrays for predictions, probabilities, and true labels
    for all batches in the given fold directory.
    """
    pred_labels = []
    true_labels = []
    probs = []

    # Iterate over files in the fold directory
    for file_name in os.listdir(fold_dir):
        if file_name.startswith("pred_labels_batch_") and file_name.endswith(".npz"):
            pred_labels.append(np.load(os.path.join(fold_dir, file_name))["arr_0"])
        elif file_name.startswith("probs_batch_") and file_name.endswith(".npz"):
            probs.append(np.load(os.path.join(fold_dir, file_name))["arr_0"])
        elif file_name.startswith("true_labels_batch_") and file_name.endswith(".npz"):
            true_labels.append(np.load(os.path.join(fold_dir, file_name))["arr_0"])

    # Concatenate all batches into single arrays
    y_pred_all = np.concatenate(pred_labels, axis=0)  # Concatenate along batch axis
    y_true_all = np.concatenate(true_labels, axis=0)  # Concatenate along batch axis
    y_prob_all = np.concatenate(probs, axis=0)  # Concatenate along batch axis

    return y_true_all, y_pred_all, y_prob_all


def plot_confusion_matrix(
    y_true, y_pred, class_names=["Background", "NCR", "ED", "ET"]
):
    # Compute confusion matrix
    cm = confusion_matrix(
        y_true.flatten(), y_pred.flatten(), labels=[0, 1, 2, 3], normalize="true"
    )

    # Plot confusion matrix as a figure
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax)
    plt.title("Confusion Matrix for Tumor Subregions")
    plt.close(fig)  # Close figure to prevent display during training
    return fig


def plot_roc_curve(y_true_all, y_prob_all, fold, epoch):
    """
    Plots ROC curves for NCR, ED, ET subregions.
    """
    class_labels = ["NCR", "ED", "ET"]
    fig, ax = plt.subplots(figsize=(10, 8))  # Create figure object for TensorBoard

    for i, class_label in enumerate(class_labels):
        # Convert to binary format for the subregion
        y_true_binary = (
            (y_true_all == i + 1).astype(int).flatten()
        )  # 1 for the subregion, 0 for others
        y_pred_probs = y_prob_all[
            :, i + 1
        ].flatten()  # Predicted probabilities for the subregion

        # Compute FPR, TPR, and AUC
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs)
        auc_score = auc(fpr, tpr)

        # Plot the ROC curve for the subregion
        ax.plot(fpr, tpr, label=f"{class_label} (AUC = {auc_score:.4f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random chance line
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves for Tumor Subregions")
    ax.legend(loc="lower right")
    plt.close(fig)

    return fig


def generate_roc_and_confusion_matrix(fold_dir, fold=1, epoch=None):
    """
    Loads predictions, probabilities, and true labels from the given fold directory,
    generates an ROC curve with AUC for NCR, ED, ET subregions, and plots a confusion matrix.
    """
    fold_dir_save = os.path.join(fold_dir, "viz")
    os.makedirs(fold_dir_save, exist_ok=True)

    # Load data
    y_true_all, y_pred_all, y_prob_all = load_fold_results(fold_dir)

    # Generate and save confusion matrix
    cm_fig = plot_confusion_matrix(
        y_true_all, y_pred_all, class_names=["Background", "NCR", "ED", "ET"]
    )
    cm_fig.savefig(
        os.path.join(fold_dir_save, f"confusion_matrix_fold_{fold}_epoch_{epoch}.png")
    )
    print(
        f"Confusion matrix saved at: {os.path.join(fold_dir_save, f'confusion_matrix_fold_{fold}_epoch_{epoch}.png')}"
    )

    # Generate and save ROC curve
    roc_fig = plot_roc_curve(y_true_all, y_prob_all, fold, epoch)
    roc_fig.savefig(
        os.path.join(fold_dir_save, f"roc_curve_fold_{fold}_epoch_{epoch}.png")
    )
    print(
        f"ROC curve saved at: {os.path.join(fold_dir_save, f'roc_curve_fold_{fold}_epoch_{epoch}.png')}"
    )

    print("Confusion matrix and ROC curve generated.")


fold_dir = r"/mnt/c/Users/agata/Desktop/thesis_tumor_segmentation/src/train/outputs/swinunetr/fold_5_results"

# Generate ROC and confusion matrix for the last epoch of the fold
generate_roc_and_confusion_matrix(fold_dir, fold=5, epoch=config.max_epochs)
