from base_visualizer import ModelVisualizerBase
from viz_utils import resize_to_fixed_size, resize_segmentation, most_disagreeing_slice, get_tumor_bbox
import matplotlib.pyplot as plt
import os 

class PredictionVisualizer(ModelVisualizerBase):
    def create_comparison_figure(self, patients, models, save_path=None):
        """
        Create a comparison figure for multiple patients and models.
        
        Args:
            patients (list): List of patient IDs
            models (list): List of model names to compare
            save_path (str, optional): Path to save the figure
        """
        # Create figure with 2 rows and len(models) + 2 columns (1 for patient names, len(models) for models, 1 for GT, 1 for FLAIR)
        fig = plt.figure(figsize=(32, 10))
        gs = plt.GridSpec(2, len(models) + 3, figure=fig, height_ratios=[1, 1], 
                         width_ratios=[0.1] + [1] * (len(models) + 2))
        
        # Create custom colormap for segmentation regions
        from matplotlib.colors import ListedColormap
        colors = ['black', 'red', 'yellow', 'blue']  # Background, NCR, ED, ET
        custom_cmap = ListedColormap(colors)
        
        # Fixed size for all images
        target_size = (256, 256)
        
        # Process each patient
        for row, patient_id in enumerate(patients):
            # Add patient name in first column
            ax_label = fig.add_subplot(gs[row, 0])
            ax_label.text(0.5, 0.5, patient_id, 
                         ha="center", va="center", 
                         fontsize=20, fontweight='bold',
                         rotation=90,
                         transform=ax_label.transAxes)
            ax_label.axis("off")
            
            # Get data for all models
            all_data = []
            patient_paths = {
                m: os.path.join(self.outputs_root, patient_id, m)
                for m in models
            }
            best_z, _ = most_disagreeing_slice(patient_id, models, patient_paths)
            for model in models:
                pred_seg_data = self.load_prediction(model, patient_id)
                seg_data      = self.load_ground_truth(patient_id)
                flair_data    = self.load_flair(patient_id)
                slice_idx     = best_z
                all_data.append((pred_seg_data, seg_data, flair_data, slice_idx))
            
            # Get ground truth slice and its bounding box for consistent cropping
            gt_slice = all_data[0][1][:, :, all_data[0][3]]
            bbox = get_tumor_bbox(gt_slice, padding=40)
            
            if bbox is None:
                print(f"No tumor found in ground truth for patient {patient_id}")
                continue
                
            y_min, y_max, x_min, x_max = bbox
            
            # Plot predictions for each model
            for col, (model, (pred_seg_data, _, flair_data, slice_idx)) in enumerate(zip(models, all_data)):
                ax = fig.add_subplot(gs[row, col + 1])  # +1 to skip patient name column
                
                # Extract and crop slices
                flair_slice = flair_data[:, :, slice_idx]
                pred_seg_slice = pred_seg_data[:, :, slice_idx]
                
                flair_crop = flair_slice[y_min:y_max, x_min:x_max]
                pred_seg_crop = pred_seg_slice[y_min:y_max, x_min:x_max]
                
                # Resize cropped images to target size
                flair_resized = resize_to_fixed_size(flair_crop, target_size)
                pred_seg_resized = resize_segmentation(pred_seg_crop, target_size)
                
                ax.imshow(flair_resized, cmap='gray')
                ax.imshow(pred_seg_resized, cmap=custom_cmap, alpha=0.5, vmin=0, vmax=3)
                
                # Add title with metrics if requested
                if row == 0:  # Only add titles for the first row
                    title = self.model_names[model]
                    ax.set_title(title, fontsize=20, fontweight='bold', pad=10)
                ax.axis('off')
                ax.set_aspect('equal')
            
            # Plot ground truth
            ax = fig.add_subplot(gs[row, len(models) + 1])  # Ground truth is second to last column
            flair_slice = all_data[0][2][:, :, all_data[0][3]]
            seg_slice = all_data[0][1][:, :, all_data[0][3]]
            
            # Crop and resize
            flair_crop = flair_slice[y_min:y_max, x_min:x_max]
            seg_crop = seg_slice[y_min:y_max, x_min:x_max]
            
            flair_resized = resize_to_fixed_size(flair_crop, target_size)
            seg_resized = resize_segmentation(seg_crop, target_size)
            
            ax.imshow(flair_resized, cmap='gray')
            ax.imshow(seg_resized, cmap=custom_cmap, alpha=0.5, vmin=0, vmax=3)
            if row == 0:  # Only add titles for the first row
                ax.set_title(self.model_names["gt"], fontsize=20, fontweight='bold', pad=10)
            ax.axis('off')
            ax.set_aspect('equal')
            
            # Plot raw FLAIR
            ax = fig.add_subplot(gs[row, len(models) + 2])  # FLAIR is last column
            flair_slice = all_data[0][2][:, :, all_data[0][3]]
            flair_crop = flair_slice[y_min:y_max, x_min:x_max]
            flair_resized = resize_to_fixed_size(flair_crop, target_size)
            
            ax.imshow(flair_resized, cmap='gray')
            if row == 0:  # Only add titles for the first row
                ax.set_title(self.model_names["flair"], fontsize=20, fontweight='bold', pad=10)
            ax.axis('off')
            ax.set_aspect('equal')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()