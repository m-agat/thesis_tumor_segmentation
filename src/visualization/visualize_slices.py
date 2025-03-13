import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch


# Load MRI modalities and segmentation data
def load_nifti(path):
    return nib.load(path).get_fdata()


data_path = (
    "/home/agata/Desktop/thesis_tumor_segmentation/data/brats2021challenge/TrainingData"
)
participants = os.listdir(data_path)

for part in participants:
    participant_folder = os.path.join(data_path, part)
    modalities = os.listdir(participant_folder)
    output_path = f"/home/agata/Desktop/thesis_tumor_segmentation/figures/Brain_Visualizations/{part}_viz.png"

    if os.path.exists(output_path):
        print(f"Visualization for {part} already exists, skipping...")
        continue

    flair_path = os.path.join(
        participant_folder, [file for file in modalities if "flair" in file][0]
    )
    t1_path = os.path.join(
        participant_folder,
        [file for file in modalities if "t1" in file and "t1ce" not in file][0],
    )
    t1ce_path = os.path.join(
        participant_folder, [file for file in modalities if "t1ce" in file][0]
    )
    t2_path = os.path.join(
        participant_folder, [file for file in modalities if "t2" in file][0]
    )
    segmentation_path = os.path.join(
        participant_folder, [file for file in modalities if "seg" in file][0]
    )

    flair = load_nifti(flair_path)
    t1 = load_nifti(t1_path)
    t1ce = load_nifti(t1ce_path)
    t2 = load_nifti(t2_path)
    segmentation = load_nifti(segmentation_path)

    # Function to find the slice with the highest non-zero pixels
    def get_max_nonzero_slice(image):
        nonzero_counts = [
            np.count_nonzero(image[:, :, i]) for i in range(image.shape[2])
        ]
        return np.argmax(nonzero_counts)

    # # Get the slice index with the maximum non-zero pixels for each modality
    slice_index = get_max_nonzero_slice(
        segmentation
    )  # same slice index used across modalities for consistency

    # # Plotting settings for high-resolution figure
    plt.figure(figsize=(20, 10), dpi=300)

    colors = {1: "green", 2: "pink", 4: "blue"}  # NCR, ED, ET
    labels = {1: "NCR (Necrotic Core)", 2: "ED (Edema)", 4: "ET (Enhancing Tumor)"}

    # # Titles for the subplots
    titles = ["FLAIR", "T1", "T1CE", "T2", "Segmentation"]

    for i, (image, title) in enumerate(
        zip([flair, t1, t1ce, t2, segmentation], titles)
    ):
        plt.subplot(1, 5, i + 1)
        plt.imshow(image[:, :, slice_index], cmap="gray")

        # Overlay the segmentation in the last subplot with a custom color map
        if title == "Segmentation":
            segmentation_slice = segmentation[:, :, slice_index]
            colored_segmentation = np.zeros(
                segmentation_slice.shape + (3,), dtype=np.float32
            )

            for label, color in colors.items():
                mask = segmentation_slice == label
                if color == "green":
                    colored_segmentation[mask] = [0, 1, 0]
                elif color == "pink":
                    colored_segmentation[mask] = [1, 0.75, 0.8]
                elif color == "blue":
                    colored_segmentation[mask] = [0, 0, 1]

            plt.imshow(colored_segmentation)

        plt.axis("off")
        plt.title(title)

    legend_elements = [
        Patch(facecolor=color, edgecolor="k", label=labels[label])
        for label, color in colors.items()
    ]
    plt.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize="large",
        title="Segmentation Key",
    )

    # Save the figure with high resolution
    plt.tight_layout(pad=1.0)

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.2)
    plt.close()

    print(f"Visualization saved to {output_path}")
