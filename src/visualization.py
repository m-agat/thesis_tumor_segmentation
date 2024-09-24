import matplotlib.pyplot as plt

def plot_segmentation_with_accuracy_and_uncertainty(img, ground_truth, predicted_segmentation, accuracy_map, variance_uncertainty, dice_uncertainty, slice_num, save_path=None):
    """
    Plot original image, ground truth, predicted segmentation, accuracy map, variance-based uncertainty map, and dice-based uncertainty map.
    """
    fig, axes = plt.subplots(1, 6, figsize=(24, 6))  

    # Original Image
    axes[0].imshow(img[:, :, slice_num], cmap='gray')
    axes[0].set_title('Original Image')

    # Ground Truth Segmentation
    axes[1].imshow(ground_truth[:, :, slice_num])
    axes[1].set_title('Ground Truth')

    # Predicted Segmentation
    axes[2].imshow(predicted_segmentation[:, :, slice_num])
    axes[2].set_title('Predicted Segmentation')

    # Accuracy Map (Correct vs. Incorrect)
    im_accuracy = axes[3].imshow(accuracy_map[:, :, slice_num], cmap='gray')
    axes[3].set_title('Accuracy Map (1: Correct, 0: Incorrect)')
    
    # Variance-based Uncertainty Map
    im_variance_uncertainty = axes[4].imshow(variance_uncertainty[:, :, slice_num], cmap='jet', alpha=0.7)
    axes[4].set_title('Variance Uncertainty')

    # Dice-based Uncertainty Map
    im_dice_uncertainty = axes[5].imshow(dice_uncertainty[:, :, slice_num], cmap='jet', alpha=0.7)
    axes[5].set_title('Dice Score Uncertainty')

    # Add color bars for the uncertainty maps
    cbar_variance = fig.colorbar(im_variance_uncertainty, ax=axes[4], orientation='vertical', shrink=0.8)
    cbar_variance.set_label('Variance-based Uncertainty')

    cbar_dice = fig.colorbar(im_dice_uncertainty, ax=axes[5], orientation='vertical', shrink=0.8)
    cbar_dice.set_label('Dice Score-based Uncertainty')
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)  # Save plot instead of showing it
        print(f"Plot saved to {save_path}")
    else:
        plt.show()