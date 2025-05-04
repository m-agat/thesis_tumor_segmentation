import argparse
from evaluator import evaluate_patient, evaluate_batch


def parse_args():
    parser = argparse.ArgumentParser("OOD Evaluation CLI")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Single patient evaluation
    p1 = sub.add_parser("patient", help="Evaluate one patient segmentation")
    p1.add_argument("--pred", required=True,
                    help="Path to predicted segmentation (NIfTI file)")
    p1.add_argument("--gt", required=True,
                    help="Path to ground-truth segmentation (NIfTI file)")
    p1.add_argument("--out", required=False,
                    help="Path to save single-patient metrics (JSON or CSV)")

    # Batch evaluation
    p2 = sub.add_parser("batch", help="Evaluate all patients in a folder")
    p2.add_argument("--pred_dir", required=True,
                    help="Directory with predicted segmentations (NIfTIs named seg_<pid>.nii.gz)")
    p2.add_argument("--gt_dir", required=True,
                    help="Directory with ground-truth segmentations (seg_<pid>.nii.gz)")
    p2.add_argument("--out_csv", required=True,
                    help="Path to save batch metrics CSV")
    p2.add_argument("--out_json", required=False,
                    help="Path to save aggregated metrics JSON")
    p2.add_argument("--patients", nargs='*', default=None,
                    help="List of patient IDs to include (default: all found)")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "patient":
        metrics = evaluate_patient(
            pred_path=args.pred,
            gt_path=args.gt
        )
        if args.out:
            evaluate_patient.save_metrics(metrics, args.out)
        else:
            print(metrics)
    else:  # batch mode
        df, summary = evaluate_batch(
            pred_dir=args.pred_dir,
            gt_dir=args.gt_dir,
            patient_ids=args.patients
        )
        df.to_csv(args.out_csv, index=False)
        print(f"[INFO] Saved per-patient metrics to {args.out_csv}")
        if args.out_json:
            evaluate_batch.save_summary(summary, args.out_json)
            print(f"[INFO] Saved aggregated metrics to {args.out_json}")

if __name__ == "__main__":
    main()
