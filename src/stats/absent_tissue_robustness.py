from stats.data import load_ground_truth, load_model_performance, merge_with_gt
from stats.utils import compute_absence_performance

def main():
    gt = load_ground_truth(r'\Users\agata\Desktop\thesis_tumor_segmentation\EDA\brats2021_class_presence_all.csv')
    model_paths = {
        'V-Net': '../models/performance/vnet/patient_metrics_test_vnet.csv',
        'Attention UNet': '../models/performance/attunet/patient_metrics_test_attunet.csv',
        'SegResNet':'../models/performance/segresnet/patient_metrics_test_segresnet.csv',
        'SwinUNETR':'../models/performance/swinunetr/patient_metrics_test_swinunetr.csv',
    }

    merged = {}
    for name, p in model_paths.items():
        perf = load_model_performance(p)
        merged[name] = merge_with_gt(perf, gt)

    regions = ['NCR present', 'ED present', 'ET present']
    print("Model robustness on absent tissue:")
    for region in regions:
        print(f"\n=== {region.split()[0]} ===")
        for name, df in merged.items():
            mean_dice, count = compute_absence_performance(df, region)
            print(f"{name}: n={count}, mean Dice={mean_dice:.3f}")