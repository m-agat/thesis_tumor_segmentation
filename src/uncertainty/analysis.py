import argparse
from pathlib import Path
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from uncertainty.constants import FIGURES_DIR, ECE_BINS, RESULTS_DIR, SUBREGIONS, UNCERTAINTY_PATTERN, PREDICTION_PATTERN, GROUND_TRUTH_PATTERN, PREDICTIONS_DIR, GROUND_TRUTH_DIR, RELIABILITY_BINS, COVERAGE_FRACTIONS
from uncertainty.ece_estimation import compute_ece_per_subregion
from uncertainty.data import load_region_maps, load_error_uncertainty_data, load_conf_correct_by_class
from uncertainty.metrics import summarize_ece, extract_region_error_uncertainty, spearman_error_uncertainty, bin_average_error, compute_reliability_stats, compute_risk_coverage 
from uncertainty.plotting import (
    plot_ece_distributions, plot_ece_summary,
    plot_binned_error_curves, plot_reliability_diagram, plot_risk_coverage_curve
)
from scipy.stats import spearmanr

region_markers = ["o", "s", "^", "D", "v", "<", ">"]
region_styles = {
    region: {"marker": region_markers[i % len(region_markers)]}
    for i, region in enumerate(SUBREGIONS)
}

def run_ece_comparison(args):
    # 1) compute ECE per‐patient and per‐region, on the fly
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    model_dirs = []
    if args.tta_dir:
        model_dirs.append(("TTA", args.tta_dir))
    if args.ttd_dir:
        model_dirs.append(("TTD", args.ttd_dir))
    if args.hybrid_dir:
        model_dirs.append(("Hybrid", args.hybrid_dir))
    if not model_dirs:
        raise RuntimeError(
            "You must specify at least one model directory: --tta-dir, --ttd-dir or --hybrid-dir"
        )
    
    rows = []
    for model_label, pred_dir in model_dirs:
        print(model_label, pred_dir)
        for pid, gt_vol, prob_vol, _ in load_region_maps(
            pred_dir,
            args.gt_dir,
            args.pred_pattern,
            args.gt_pattern,
            SUBREGIONS
        ):  
            ece_dict = compute_ece_per_subregion(
                probabilities=prob_vol,
                gt=gt_vol,
                subregions=SUBREGIONS,
                num_bins=ECE_BINS,
            )
        
            for region, ece in ece_dict.items():
                rows.append({
                    "MODEL":   model_label,
                    "PATIENT": pid,
                    "REGION":  region,
                    "ECE":     ece,
                })

    df = pd.DataFrame(rows)

    # 2) summarize the findings
    summary = summarize_ece(df, regions=args.regions) 

    # 3) make plots
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
    plot_ece_distributions(df, args.regions, ax1)
    plot_ece_summary(summary, args.regions, ax2)

    out_path = FIGURES_DIR / "ece_comparison.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison figure to {out_path}")


def run_error_uncertainty(args):
    # ensure output dirs exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # derive a label for this model from the pred-dir name
    model_label = args.pred_dir.name

    # build patterns for per-region uncertainty files
    unc_pats = { r: f"uncertainty_{r}*.nii.gz" for r in SUBREGIONS }

    # initialize storage
    all_bins = { r: [] for r in SUBREGIONS }
    all_avg  = { r: [] for r in SUBREGIONS }

    # load and compute per-patient
    for pid, gt, prob, uncs in load_error_uncertainty_data(
        args.pred_dir,
        args.gt_dir,
        args.pred_pattern,
        unc_pats,
        args.gt_pattern,
    ):  
        for region, label in SUBREGIONS.items():
            err, uni = extract_region_error_uncertainty(prob, gt, uncs[region], label)
            rho, pval = spearman_error_uncertainty(err, uni)
            print(f"{model_label} {region}: SpearmanR={rho:.3f}, p={pval:.3g}")

            centers, avg_err = bin_average_error(err, uni, args.unc_bins)
            all_bins[region] = centers
            all_avg[region]  = avg_err

    # plot one combined figure per model
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_binned_error_curves(all_bins, all_avg, ax, region_styles)
    out_path = FIGURES_DIR / f"{model_label}_uncertainty_error.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {model_label} uncertainty-error plot to {out_path}")


def run_reliability(args: argparse.Namespace):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # build mapping from integer label → region name
    class_labels = {label: name for name, label in SUBREGIONS.items()}

    # load data: yields (label, confidences, correctness)
    generator = load_conf_correct_by_class(
        args.pred_dir,
        args.gt_dir,
        args.pred_pattern,
        args.gt_pattern,
        class_labels=class_labels
    )

    # prepare subplots grid
    n = len(class_labels)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()

    # iterate one panel per region
    for ax, (label, conf, corr) in zip(axes, generator):
        region_name = class_labels[label]
        stats = compute_reliability_stats(conf, corr, args.bins)
        plot_reliability_diagram(stats, region_name, ax)

    # hide any extra axes
    for ax in axes[n:]:
        ax.set_visible(False)

    out_file = FIGURES_DIR / "reliability_diagrams.png"
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved reliability diagrams to {out_file}")

def run_risk_coverage(args: argparse.Namespace):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # accumulate across all patients
    unc_pats = { r: f"uncertainty_{r}*.nii.gz" for r in SUBREGIONS }
    vox_data = { r: {"error": [], "unc": []} for r in SUBREGIONS }
    for pid, gt, prob, uncs in load_error_uncertainty_data(
        args.pred_dir,
        args.gt_dir,
        args.pred_pattern,
        unc_pats,
        args.gt_pattern
    ):
        for region, label in SUBREGIONS.items():
            mask = gt == label
            if not mask.any():
                continue
            err = -np.log(prob[label][mask] + args.eps)
            vox_data[region]["error"].append(err)
            vox_data[region]["unc"].append(uncs[region][mask])

    fig, ax = plt.subplots(figsize=(10, 6))
    for region, data in vox_data.items():
        errors = np.concatenate(data["error"])
        uncs   = np.concatenate(data["unc"])
        cov, risk = compute_risk_coverage(
            uncs, errors, np.array(args.coverage_fractions)
        )
        plot_risk_coverage_curve(
            ax, cov, risk, region,
            marker=region_styles[region]["marker"]
        )
        rho, p = spearmanr(errors, uncs)
        print(f"Region {region}: SpearmanR={rho:.4f} (p={p:.4g})")

    out_file = FIGURES_DIR / f"{args.model}_risk_coverage.png"
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved risk-coverage plot to {out_file}")

def main():
    parser = argparse.ArgumentParser(prog="uncertainty")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- ece subcommand ---
    p_ece = sub.add_parser("ece", help="Compute & compare ECE across models")
    p_ece.add_argument("--tta-dir",    type=Path, default=None,
                       help="Directory with TTA softmax volumes")
    p_ece.add_argument("--ttd-dir",    type=Path, default=None,
                       help="Directory with TTD softmax volumes")
    p_ece.add_argument("--hybrid-dir", type=Path, default=None,
                       help="Directory with Hybrid softmax volumes")
    p_ece.add_argument("--gt-dir",     type=Path, default=GROUND_TRUTH_DIR,
                       help="Directory with ground-truth segmentations")
    p_ece.add_argument("--pred-pattern", type=str, default=PREDICTION_PATTERN,
                       help="Glob pattern to match softmax files (e.g. '*_softmax.nii.gz')")
    p_ece.add_argument("--gt-pattern",   type=str, default=GROUND_TRUTH_PATTERN,
                       help="Glob pattern to match GT files (e.g. '*_seg.nii.gz')")
    p_ece.add_argument(
        "--regions", nargs="+",
        default=["NCR","ED","ET"],
        help="List of region names"
    )
    p_ece.set_defaults(func=run_ece_comparison)

    # --- error-uncertainty subcommand ---
    p_err = sub.add_parser("error-uncertainty", help="Correlate error with uncertainty")
    p_err.add_argument("--pred-dir", type=Path, required=True)
    p_err.add_argument("--gt-dir",   type=Path, default=GROUND_TRUTH_DIR)
    p_err.add_argument("--out-csv",  default="error_unc_correlation.csv")
    p_err.add_argument("--unc-bins",     type=int, default=20,
                       help="Number of bins for uncertainty")
    p_err.add_argument("--plot",     action="store_true",
                       help="Generate per-region scatter & binned plots")
    p_err.set_defaults(func=run_error_uncertainty)

    # --- reliability sub-command ---
    p_rel = sub.add_parser(
        "reliability",
        help="Generate per-class reliability diagrams"
    )
    p_rel.add_argument(
        "--pred-dir",
        type=Path,
        default=PREDICTIONS_DIR,
        help="Directory of softmax prediction volumes"
    )
    p_rel.add_argument(
        "--gt-dir",
        type=Path,
        default=GROUND_TRUTH_DIR,
        help="Directory of ground-truth segmentation volumes"
    )
    p_rel.add_argument(
        "--pred-pattern",
        type=str,
        default=PREDICTION_PATTERN.replace("{id}", "*"),
        help="Glob pattern to match softmax files (e.g. '*_softmax.nii.gz')"
    )
    p_rel.add_argument(
        "--gt-pattern",
        type=str,
        default=GROUND_TRUTH_PATTERN.replace("{id}", "*"),
        help="Glob pattern to match GT files (e.g. '*_seg.nii.gz')"
    )
    p_rel.add_argument(
        "--bins",
        type=int,
        default=RELIABILITY_BINS,
        help="Number of bins to use in the reliability diagram"
    )
    p_rel.set_defaults(func=run_reliability)

    # --- risk-coverage subcommand ---
    p_rc = sub.add_parser("risk-coverage",
        help="Compute & plot risk–coverage curves"
    )
    p_rc.add_argument("--pred-dir",   type=Path,
        default=PREDICTIONS_DIR,
        help="Directory of softmax volumes"
    )
    p_rc.add_argument("--gt-dir",     type=Path,
        default=GROUND_TRUTH_DIR,
        help="Directory of ground-truth segmentations"
    )
    p_rc.add_argument("--pred-pattern", type=str,
        default=PREDICTION_PATTERN.replace("{id}", "*"),
        help="Glob for softmax files"
    )
    p_rc.add_argument("--gt-pattern",   type=str,
        default=GROUND_TRUTH_PATTERN.replace("{id}", "*"),
        help="Glob for GT files (use '{id}' placeholder)"
    )
    p_rc.add_argument("--model",       type=str, required=True,
        help="Model name (for output filename)"
    )
    p_rc.add_argument("--coverage-fractions", type=float, nargs="+",
        default=list(COVERAGE_FRACTIONS),
        help="Fractions of voxels to keep (0-1)"
    )
    p_rc.add_argument("--eps", type=float, default=1e-8,
        help="Small epsilon to avoid log(0)"
    )
    p_rc.set_defaults(func=run_risk_coverage)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
