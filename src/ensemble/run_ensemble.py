# run_ensemble.py
import os, torch
from ensemble_utils import *
from segmenters import SimpleAverage, PerfWeighted, TTAWeighted, TTDWeighted, HybridWeighted
from config.config import device, roi, sw_batch_size, infer_overlap, model_paths, json_path, root_dir, args
from dataset.dataloaders import load_test_data
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models.models as models


MODEL_MAP = {
  "swinunetr": (models.swinunetr_model, model_paths["swinunetr"]),
  "segresnet": (models.segresnet_model, model_paths["segresnet"]),
  "attunet":   (models.attunet_model,   model_paths["attunet"]),
}

def main(method, out_dir, n_iter=10, pid=None):
    """
    Run ensemble inference and save results.

    Args:
        method (str): Fusion strategy. One of:
            - "simple": simple average of logits
            - "perf": performance-weighted logits
            - "tta": test-time augmentation + perf weighting
            - "ttd": test-time dropout + perf weighting
            - "hybrid": hybrid TTA + TTD uncertainty weighting
        out_dir (str): Directory where outputs (softmax maps, segmentations,
            uncertainties, and metrics) will be saved.
        n_iter (int, optional): Number of Monte Carlo iterations for TTA/TTD
            strategies. Default is 10. Only used when `method` is "tta", "ttd",
            or "hybrid".
        pid (str, optional): If provided, restrict inference to this one patient ID.
            Patient IDs are parsed from the file path using digits in the
            filename. Default is None (process all patients).
    """
    print("[INFO] Loading test dataset...")
    loader = load_test_data(json_path, root_dir)
    print("[INFO] Initializing models and inferers...")
    models_dict = load_all_models(MODEL_MAP, device, roi, sw_batch_size, infer_overlap)
    print(f"[INFO] Loaded models: {list(models_dict.keys())}")

    # compute perf weights once
    print("[INFO] Computing performance metrics and weights...")
    metrics_map = {m: load_weights(f"../models/performance/{m}/average_metrics.json")
                   for m in models_dict}
    perf_scores = {
      r: {m: compute_composite_scores(metrics_map[m], WTS)[r] for m in models_dict}
      for r in REGIONS
    }
    perf_weights = normalize_weights(perf_scores)
    print("[INFO] Performance weights computed for regions: ", REGIONS)

    # choose segmenter
    segmenter = {
      "simple": SimpleAverage(models_dict, perf_weights, device),
      "perf":   PerfWeighted(models_dict, perf_weights, device),
      "tta":    TTAWeighted(models_dict, perf_weights, device, n_iter),
      "ttd":    TTDWeighted(models_dict, perf_weights, device, n_iter),
      "hybrid": HybridWeighted(models_dict, perf_weights, device, n_iter),
    }[method]
    print(f"[INFO] Using fusion strategy: {segmenter.__class__.__name__}")

    os.makedirs(out_dir, exist_ok=True)
    patient_metrics = []

    with torch.no_grad():
        data_iter = loader if pid is None else config.find_patient_by_id(pid, loader)
        for batch in data_iter:
            img = batch["image"].to(device)
            ref = batch["path"][0]
            gt  = batch["label"].to(device)

            fused_logits, fused_unc = segmenter.fuse(img)
            probs, seg = segmenter.postprocess(fused_logits)
            pred_list, gt_list = segmenter.prepare_for_eval(seg, gt)

            # compute metrics & save
            dice, hd95, sens, spec = compute_metrics(pred_list, gt_list)
            pid_str = extract_patient_id(ref)
            print(f"[INFO] Processing patient {pid_str}")
            patient_metrics.append({
                'patient_id': pid_str,
                **{f"Dice {REGIONS[i]}": dice[i] for i in range(len(REGIONS)-1)},
                **{f"HD95 {REGIONS[i]}": hd95[i] for i in range(len(REGIONS)-1)},
                **{f"Sensitivity {REGIONS[i]}": sens[i] for i in range(len(REGIONS)-1)},
                **{f"Specificity {REGIONS[i]}": spec[i] for i in range(len(REGIONS)-1)},
            })
            print(f"[INFO] Metrics for patient {pid_str}: Dice={dice.tolist()}, HD95={hd95.tolist()}, Sensitivity={sens.tolist()}, Specificity={spec.tolist()}")


            save_nifti(probs.cpu(), ref, os.path.join(out_dir,f"softmax_{pid_str}.nii.gz"), np.float32)
            save_nifti(seg.cpu(), ref, os.path.join(out_dir,f"seg_{pid_str}.nii.gz"))
            if fused_unc:
               for r,unc_map in fused_unc.items():
                   save_nifti(unc_map, ref, os.path.join(out_dir,f"unc_{r}_{pid_str}.nii.gz"), np.float32)
            print(f"[INFO] Outputs saved for patient {pid_str}")
    print("[INFO] Saving aggregated metrics...")
    save_metrics(patient_metrics, out_dir)
    print("[INFO] Ensemble inference completed successfully.")

if __name__ == "__main__":
    main(
        method  = args.method,
        out_dir = args.output_path,
        n_iter  = args.n_iter,
        pid     = args.patient_id,
    )