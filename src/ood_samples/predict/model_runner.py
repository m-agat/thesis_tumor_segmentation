import glob
import torch
import numpy as np
from monai import data
import os, sys
from pathlib import Path

# get the directory of this script:
script_dir = Path(__file__).resolve().parent
# go up three levels: predict → ood_samples → src
src_root = script_dir.parents[2]  
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from config.config       import model_paths, device, roi, sw_batch_size, infer_overlap, batch_size
from ensemble.ensemble_utils  import load_all_models, load_weights, compute_composite_scores, normalize_weights, extract_patient_id, REGIONS, compute_metrics, save_nifti  
from ensemble.segmenters      import SimpleAverage, PerfWeighted, TTAWeighted, TTDWeighted, HybridWeighted
import models.models as models
from dataset.transforms import get_test_transforms
from ood_samples.utils.io import load_nifti, OOD_DATA_PATH, WTS


MODEL_MAP = {
  "swinunetr": (models.swinunetr_model, model_paths["swinunetr"]),
  "segresnet": (models.segresnet_model, model_paths["segresnet"]),
  "attunet":   (models.attunet_model,   model_paths["attunet"]),
}

def load_predict_models():
    """
    Load each model and its sliding-window inferer onto the target device.
    Returns a dict: {model_name: (model, inferer_fn)}
    """
    return load_all_models(MODEL_MAP, device, roi, sw_batch_size, infer_overlap)


def create_data_loader(mri_scans, pid):
    """
    Build a MONAI DataLoader from a list of preprocessed MRI scan paths.
    """

    # modality ordering
    modality_order = {'flair': 0, 't1ce': 1, 't1': 2, 't2': 3}
    scans = sorted(
        mri_scans,
        key=lambda x: modality_order.get(
            next((m for m in modality_order if m in x.lower()), None), 99
        )
    )
    print(f"[INFO] Loaded MRI scans for {pid}: {[os.path.basename(f) for f in scans]}")

    # ground truth label path
    gt_path = os.path.join(OOD_DATA_PATH, pid, f"{pid}_seg.nii.gz")
    entry = {"image": scans, "label": gt_path, "path": scans[0]}
    ds = data.Dataset(data=[entry], transform=get_test_transforms())
    loader = data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    return loader


def run_predictions(loader, models_dict, methods, out_base, n_iter=10):
    """
    For each fusion method, run the ensemble on all cases and save segmentations.

    Args:
      loader: DataLoader over preprocessed cases
      models_dict: output of load_predict_models()
      methods: list of methods in ['simple','perf','tta','ttd','hybrid']
      out_base: base directory to create per-method subfolders
      n_iter: MC iterations for TTA/TTD methods
    """
    # precompute performance weights
    perf_base = os.path.join(src_root, "src", "models", "performance")

    # Load each model’s JSON file, error if missing
    metrics_map = {}
    for m in models_dict:
        fp = os.path.join(perf_base, m, "average_metrics.json")
        w = load_weights(fp)
        if w is None:
            raise FileNotFoundError(f"Missing performance file for {m}: {fp}")
        metrics_map[m] = w

    perf_scores = {
        r: {m: compute_composite_scores(metrics_map[m], WTS)[r] for m in models_dict}
        for r in REGIONS
    }
    perf_weights = normalize_weights(perf_scores)

    # map method -> segmenter instance
    segmenters = {
        "simple": SimpleAverage(models_dict, perf_weights, device),
        "perf":   PerfWeighted(models_dict, perf_weights, device),
        "tta":    TTAWeighted(models_dict, perf_weights, device, n_iter),
        "ttd":    TTDWeighted(models_dict, perf_weights, device, n_iter),
        "hybrid": HybridWeighted(models_dict, perf_weights, device, n_iter),
    }

    for method in methods:
        segmenter = segmenters[method]
        out_dir = os.path.join(out_base, method)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[INFO] Running method '{method}', output -> {out_dir}")

        with torch.no_grad():
            for batch in loader:
                img = batch["image"].to(device)
                ref = batch["path"][0]
                gt = batch["label"].to(device)

                pid = extract_patient_id(ref)
                print(f"[INFO] Processing patient {pid}")

                # fuse logits and postprocess
                fused_logits, fused_unc = segmenter.fuse(img)
                probs, seg = segmenter.postprocess(fused_logits)
                
                gt_path = os.path.join(OOD_DATA_PATH, pid, f"{pid}_seg.nii.gz")
                if os.path.exists(gt_path):
                    # compute metrics & save
                    gt_data, _, _ = load_nifti(gt_path)
                    gt = torch.from_numpy(gt_data).to(device)
                    pred_list, gt_list = segmenter.prepare_for_eval(seg, gt)
                    dice, hd95, sens, spec = compute_metrics(pred_list, gt_list)
                    print(f"[INFO] Metrics for patient {pid}: Dice={dice.tolist()}, HD95={hd95.tolist()}, Sensitivity={sens.tolist()}, Specificity={spec.tolist()}")

                # save segmentation
                soft_out = os.path.join(out_dir, f"softmax_{pid}.nii.gz")
                seg_out = os.path.join(out_dir, f"seg_{pid}.nii.gz")
                save_nifti(seg.cpu().numpy().astype(np.uint8), ref, soft_out)
                save_nifti(probs.cpu().numpy().astype(np.uint8), ref, seg_out)
                if fused_unc:
                    for r,unc_map in fused_unc.items():
                        save_nifti(unc_map, ref, os.path.join(out_dir,f"unc_{r}_{pid}.nii.gz"), np.float32)

                print(f"[INFO] Saved {seg_out}")

    print("[INFO] All predictions complete.")