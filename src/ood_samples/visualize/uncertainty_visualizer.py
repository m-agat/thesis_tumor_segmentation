from base_visualizer import ModelVisualizerBase
from viz_utils import resize_to_fixed_size, most_disagreeing_slice, get_tumor_bbox, resize_segmentation
import matplotlib.pyplot as plt
import os 
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from viz_utils import *

class UncertaintyVisualizer(ModelVisualizerBase):
    def load_uncertainty(self, model_name: str, patient_id: str, region: str):
        """
        Load the (already saved) voxel-wise uncertainty map for a given model,
        patient and tumour sub-region (ED, ET, NCR).

        Expected file name:  uncertainty_<REGION>_<PID>.nii.gz
        """
        path = self.get_model_path(model_name, patient_id)
        fname = f"uncertainty_{region}_{patient_id}.nii.gz"
        return load_nifti(os.path.join(path, fname))
    
    def create_uncertainty_figure(self,
                              patients: list[str],
                              models:   list[str],
                              regions:  list[str] = ("NCR", "ED", "ET"),
                              save_path: str | None = None):
        """
        For every patient create one figure:
            rows   = models (e.g. TTD, TTA, Hybrid)
            cols   = tumour sub-regions (NCR, ED, ET)
            cells  = uncertainty overlay on FLAIR
        """
        target_size = (256, 256)
        unc_cmap    = get_cmap("hot")
        norm        = Normalize(vmin=0.0, vmax=1.0)     # used for colour-bar

        for pid in patients:
            # -------- choose slice & bounding box --------------------------------
            patient_paths = {
                m: os.path.join(self.outputs_root, pid, m)
                for m in models
            }
            best_z, _ = most_disagreeing_slice(pid, models, patient_paths)
            flair_vol = self.load_flair(pid)
            gt_vol    = self.load_ground_truth(pid)
            gt_slice  = gt_vol[:, :, best_z]
            bbox      = get_tumor_bbox(gt_slice, padding=40) \
                        or (0, gt_slice.shape[0], 0, gt_slice.shape[1])
            y0, y1, x0, x1 = bbox

            # -------- figure & gridspec -----------------------------------------
            n_rows = len(models)          # models on the vertical axis
            n_cols = len(regions) + 1     # +1 narrow label column
            fig   = plt.figure(figsize=(3.3 * n_cols, 3.0 * n_rows))
            gs    = plt.GridSpec(n_rows, n_cols,
                                figure=fig,
                                height_ratios=[1]*n_rows,
                                width_ratios=[0.25] + [1]*len(regions))

            fig.suptitle(f"Uncertainty maps – {pid}", fontsize=22, y=1.02)

            # -------- iterate rows(models) / cols(regions) ----------------------
            for r, model in enumerate(models):
                # left-hand label cell (row titles = model names)
                ax_lbl = fig.add_subplot(gs[r, 0])
                ax_lbl.axis("off")
                ax_lbl.text(0.5, 0.5, self.model_names[model],
                            ha="center", va="center",
                            fontsize=14, fontweight="bold", rotation=90)

                # load once per model to save I/O
                unc_vols = {reg: self.load_uncertainty(model, pid, reg) for reg in regions}

                for c, region in enumerate(regions, start=1):
                    ax = fig.add_subplot(gs[r, c])

                    unc_slice = unc_vols[region][:, :, best_z][y0:y1, x0:x1]
                    flair_c   = flair_vol[:, :, best_z][y0:y1, x0:x1]

                    flair_rs  = resize_to_fixed_size(flair_c, target_size)
                    unc_rs    = resize_to_fixed_size(unc_slice, target_size)

                    # 0-1 normalisation per slice for visual contrast
                    if unc_rs.max() > unc_rs.min():
                        unc_norm = (unc_rs - unc_rs.min()) / (unc_rs.max() - unc_rs.min())
                    else:
                        unc_norm = unc_rs

                    ax.imshow(flair_rs, cmap="gray")
                    ax.imshow(unc_norm, cmap=unc_cmap, alpha=0.5, vmin=0.09, vmax=1)
                    ax.axis("off")

                    if r == 0:          # column headers (region names)
                        ax.set_title(region, fontsize=16, fontweight="bold")

            # -------- unified colour-bar ----------------------------------------
            cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])     # [left, bottom, w, h]
            sm  = plt.cm.ScalarMappable(cmap=unc_cmap, norm=norm)
            sm.set_array([])
            cb  = fig.colorbar(sm, cax=cax)
            cb.set_label("Uncertainty", rotation=270, labelpad=15, fontsize=12)

            plt.subplots_adjust(wspace=0.05, hspace=0.05)

            # -------- save / show ----------------------------------------------
            if save_path:
                base, ext = os.path.splitext(save_path)
                ext = ext if ext else ".png"
                out = f"{base}_{pid}{ext}" if ext else f"{save_path}.png"
                plt.savefig(out, dpi=200, bbox_inches="tight")
            plt.show()
            plt.close(fig)