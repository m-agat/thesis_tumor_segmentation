import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.utils import (
    load_model,
    extract_patient_id,
    save_segmentation_as_nifti,
    compute_metrics,
)
from dataset.dataloaders import load_test_data
import models.models as model_defs
import config.config as cfg

def predict_single_model(
    model: torch.nn.Module,
    inferer,
    test_loader: DataLoader,
    output_dir: Path,
    pid: str
):
    """
    Run inference on all cases in test_loader, save segmentations & print metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.no_grad():
        data_iter = test_loader if pid is None else cfg.find_patient_by_id(pid, test_loader)
        for batch in test_loader:
            image = batch["image"].to(cfg.device)
            gt     = batch["label"].to(cfg.device)
            path   = batch["path"][0]
            pid    = extract_patient_id(path)

            # forward
            logits = inferer(image).squeeze(0)
            seg    = torch.softmax(logits, dim=0).argmax(dim=0, keepdim=True)

            # build one-hot for metrics
            pred_oh = torch.nn.functional.one_hot(seg, num_classes=4).permute(3,0,1,2).float()
            if gt.shape[1] == 4:
                gt_oh = gt  # already one-hot
            else:
                gt_oh = torch.nn.functional.one_hot(gt.long(), num_classes=4).permute(1,0,2,3,4).float()

            # metrics
            dice, hd95, sens, spec = compute_metrics(pred_oh, gt_oh)

            # print metrics exactly as before
            print(f"\nPatient {pid} — Model {model_defs.__name__}\n")
            print(f"Dice:       BG {dice[0].item():.4f}, NCR {dice[1].item():.4f}, ED {dice[2].item():.4f}, ET {dice[3].item():.4f}")
            print(f"HD95:       BG {hd95[0].item():.2f}, NCR {hd95[1].item():.2f}, ED {hd95[2].item():.2f}, ET {hd95[3].item():.2f}")
            print(f"Sensitivity:BG {sens[0].item():.4f}, NCR {sens[1].item():.4f}, ED {sens[2].item():.4f}, ET {sens[3].item():.4f}")
            print(f"Specificity:BG {spec[0].item():.4f}, NCR {spec[1].item():.4f}, ED {spec[2].item():.4f}, ET {spec[3].item():.4f}\n")

            # save
            out_path = output_dir / f"seg_{pid}.nii.gz"
            save_segmentation_as_nifti(seg, path, str(out_path))

    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(
        description="Batch-predict and evaluate segmentation models"
    )
    p.add_argument("--model", "-m",
                   choices=cfg.model_paths.keys(),
                   required=True,
                   help="Which model to run")
    p.add_argument("--patient", "-p", default=None,
                   help="Optionally run only this patient ID")
    p.add_argument("--out", "-o", default="models/predictions",
                   help="Root output folder")
    args = p.parse_args()

    # build test loader (optionally filter by patient)
    test_loader = load_test_data(cfg.json_path, cfg.root_dir)

    # instantiate model & inferer
    model_ctor      = model_defs.__dict__[f"{args.model}_model"]
    checkpoint_path = cfg.model_paths[args.model]
    model = load_model(model_ctor, checkpoint_path, cfg.device)
    inferer = model_defs.get_inferer(model)

    out_dir = Path(args.out) / args.model
    predict_single_model(model, inferer, test_loader, out_dir, args.patient)


if __name__ == "__main__":
    main()
