import os 
import nibabel as nib
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

plt.rcParams["figure.figsize"] = (12, 4)     # width x height in inches
plt.rcParams["font.size"] = 14               # base font size
plt.rcParams["axes.titlesize"] = 16          # axis title font size
plt.rcParams["axes.labelsize"] = 14          # axis label font size
plt.rcParams["legend.fontsize"] = 12         # legend font size
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
mpl.rcParams['pdf.compression'] = 0
# Cases where one model significantly outperforms the others in Dice NCR:
#     patient_id  Best_DiceNCR    Best_Model  SecondBest_DiceNCR Second_Best_Model  Diff_DiceNCR
# 7        01440      0.457781   Seg_DiceNCR            0.096792      Swin_DiceNCR      0.360989
# 12       01556      0.777931   Att_DiceNCR            0.272482       Seg_DiceNCR      0.505449
# 23       01405      0.456522  Swin_DiceNCR            0.000000       Seg_DiceNCR      0.456522
# 29       01589      0.849716   Seg_DiceNCR            0.615527       Att_DiceNCR      0.234189
# 32       01157      0.874934   Seg_DiceNCR            0.765156      Swin_DiceNCR      0.109779
# 49       01032      0.550122   Seg_DiceNCR            0.220798       Att_DiceNCR      0.329324
# 69       01146      0.408497  Swin_DiceNCR            0.244586       Att_DiceNCR      0.163911
# 91       00025      0.411255   Seg_DiceNCR            0.000000      Swin_DiceNCR      0.411255
# 103      01661      0.654778   Att_DiceNCR            0.485417      Swin_DiceNCR      0.169361
# 125      01432      0.367124   Seg_DiceNCR            0.011716      Swin_DiceNCR      0.355408
# 152      01512      0.445671   Att_DiceNCR            0.111594      Swin_DiceNCR      0.334077
# 155      01347      0.903226  Swin_DiceNCR            0.777778       Att_DiceNCR      0.125448
# 165      00332      0.627963   Seg_DiceNCR            0.312796      Swin_DiceNCR      0.315167
# 171      01474      0.612406   Seg_DiceNCR            0.159083       Att_DiceNCR      0.453322

# Cases where one model significantly outperforms the others in Dice ED:
#     patient_id  Best_DiceED   Best_Model  SecondBest_DiceED Second_Best_Model  Diff_DiceED
# 6        01502     0.442906  Swin_DiceED           0.333211        Att_DiceED     0.109695
# 12       01556     0.702839   Seg_DiceED           0.592014        Att_DiceED     0.110825
# 13       01493     0.709648   Att_DiceED           0.530264        Seg_DiceED     0.179384
# 25       01230     0.405052   Att_DiceED           0.277973       Swin_DiceED     0.127079
# 51       00021     0.522728   Att_DiceED           0.236533        Seg_DiceED     0.286195
# 91       00025     0.650878  Swin_DiceED           0.376040        Seg_DiceED     0.274838
# 94       00742     0.774798   Att_DiceED           0.673768        Seg_DiceED     0.101030
# 127      00280     0.599075   Seg_DiceED           0.491202       Swin_DiceED     0.107872
# 144      00694     0.583603   Att_DiceED           0.329238       Swin_DiceED     0.254365
# 171      01474     0.680790   Seg_DiceED           0.508311        Att_DiceED     0.172479

# Cases where one model significantly outperforms the others in Dice ET:
#     patient_id  Best_DiceET   Best_Model  SecondBest_DiceET Second_Best_Model  Diff_DiceET
# 3        00380     0.237657   Seg_DiceET           0.106244       Swin_DiceET     0.131413
# 5        01075     0.780008   Seg_DiceET           0.677369       Swin_DiceET     0.102639
# 25       01230     0.682826   Seg_DiceET           0.561063        Att_DiceET     0.121762
# 49       01032     0.946776   Seg_DiceET           0.802454        Att_DiceET     0.144322
# 66       00028     0.604817  Swin_DiceET           0.394167        Seg_DiceET     0.210649
# 94       00742     0.779143   Seg_DiceET           0.616861       Swin_DiceET     0.162282
# 125      01432     0.677408   Seg_DiceET           0.576739       Swin_DiceET     0.100669
# 140      00338     0.903577   Seg_DiceET           0.721014       Swin_DiceET     0.182562
# 171      01474     0.803100   Seg_DiceET           0.696412       Swin_DiceET     0.106688

# Cases definition
attunet_case    = "01556"  # best at segmenting NCR
segresnet_case  = "01474"  # good at segmenting ET and ED
borderline_case = "01529"  # most challenging

# Optionally, you might want to use different slice indices per case:
slice_indices = {
    attunet_case: 60,
    segresnet_case: 75,
    borderline_case: 70  # change if needed
}

# For labeling the columns, you can add descriptive titles.
cases = [
    (attunet_case, ""),
    (segresnet_case, ""),
    (borderline_case, "")
]

gt_path = r"\\wsl.localhost\Ubuntu-22.04\home\magata\data\brats2021challenge\RelabeledTrainingData"
pred_base = "../models/predictions"

# Dictionary mapping model keys to display names.
models_dict_names = {
    "vnet": "V-Net",
    "segresnet": "SegResNet",
    "attunet": "Attention UNet",
    "swinunetr": "SwinUNETR"
}
# Define the order and names for the rows.
sources = [
    ("vnet", models_dict_names["vnet"]),
    ("segresnet", models_dict_names["segresnet"]),
    ("attunet", models_dict_names["attunet"]),
    ("swinunetr", models_dict_names["swinunetr"]),
    ("gt", "Ground Truth")
]

def create_overlay(seg_slice, overlay_colors):
    """
    Create a full overlay image (RGBA) from the 2D segmentation slice.
    
    Parameters:
    - seg_slice: a 2D numpy array containing integer labels.
    - overlay_colors: dictionary mapping label to an RGBA tuple.
    
    Returns:
    - overlay: an RGBA image of the same height and width as seg_slice.
    """
    overlay = np.zeros((seg_slice.shape[0], seg_slice.shape[1], 4), dtype=float)
    for label, color in overlay_colors.items():
        mask = seg_slice == label
        overlay[mask] = color
    return overlay

def load_t1ce_and_seg(patient_id, slice_index):
    """
    Loads the T1ce slice and ground truth segmentation slice for a given patient.
    Returns: t1ce_slice (2D numpy array), gt_seg_slice (2D numpy array)
    """
    t1ce_path = os.path.join(gt_path, f"BraTS2021_{patient_id}", f"BraTS2021_{patient_id}_t1ce.nii.gz")
    gt_seg_path = os.path.join(gt_path, f"BraTS2021_{patient_id}", f"BraTS2021_{patient_id}_seg.nii.gz")
    
    # Load T1ce image and extract slice.
    t1ce_img = nib.load(t1ce_path)
    t1ce_data = t1ce_img.get_fdata()
    t1ce_slice = t1ce_data[:, :, slice_index]
    
    # Load Ground Truth segmentation.
    gt_seg = nib.load(gt_seg_path).get_fdata().astype(np.int32)
    gt_seg_slice = gt_seg[:, :, slice_index]
    
    return t1ce_slice, gt_seg_slice

def load_prediction(patient_id, model, slice_index):
    """
    Loads the predicted segmentation for the given model and patient and extracts the required slice.
    """
    pred_seg_path = os.path.join(pred_base, model, f"{model}_{patient_id}_pred_seg.nii.gz")
    seg = nib.load(pred_seg_path).get_fdata().astype(np.int32)
    return seg[:, :, slice_index]


def visualize_with_gridspec(cases, sources, slice_indices):
    overlay_colors = {
        1: (1, 0, 0, 0.8),   
        2: (1, 1, 0, 0.8),   
        3: (0, 0, 1, 0.8)    
    }

    n_rows = len(sources)
    n_cols_cases = len(cases)
    total_cols = n_cols_cases + 1

    fig = plt.figure(figsize=(12, 8))
    
    # Make the label column narrower, e.g., 6% of the width
    width_ratios = [0.06] + [1]*n_cols_cases
    
    # Create a GridSpec with minimal wspace
    gs = GridSpec(
        nrows=n_rows, ncols=total_cols,
        figure=fig, 
        width_ratios=width_ratios,
        wspace=0.0,   # minimal horizontal space
        hspace=0.02   # minimal vertical space
    )
    
    for row, (source_key, model_label) in enumerate(sources):
        # Left label axis
        
        ax_label = fig.add_subplot(gs[row, 0], rasterized=True)
        ax_label.text(
            0.0, 0.5, model_label,  # place at x=0 so it’s flush left
            ha="left", va="center",
            fontsize=12, 
            transform=ax_label.transAxes
        )
        ax_label.axis("off")
        
        for col, (patient_id, _) in enumerate(cases, start=1):
            slice_index = slice_indices.get(patient_id, 60)
            t1ce_slice, gt_seg_slice = load_t1ce_and_seg(patient_id, slice_index)
            
            if source_key == "gt":
                seg_slice = gt_seg_slice
            else:
                seg_slice = load_prediction(patient_id, source_key, slice_index)

            overlay = create_overlay(seg_slice, overlay_colors)
            
            ax = fig.add_subplot(gs[row, col])
            # After creating each Axes, adjust its position:
            pos = ax.get_position()
            delta_w, delta_h = pos.width * 0.05, pos.height * 0.05
            # For instance, shrink the gaps by moving the left/right edges a little:
            new_pos = [pos.x0 + delta_w, pos.y0 + delta_h, pos.width - 2*delta_w, pos.height - 2*delta_h]
            ax.set_position(new_pos)
            ax.imshow(t1ce_slice, cmap="gray", interpolation="none")
            ax.imshow(overlay, interpolation="none")
            ax.axis("off")

    # Force minimal spacing around edges
    fig.subplots_adjust(
        left=0.0, right=1.0, 
        top=0.94, bottom=0.06,
        wspace=0.0, hspace=0.0
    )
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)  # tight_layout with zero padding
    plt.savefig("three_cases_composite_gridspec.png", dpi=300, bbox_inches="tight")
    plt.show()

# Create the composite figure.
visualize_with_gridspec(cases, sources, slice_indices)