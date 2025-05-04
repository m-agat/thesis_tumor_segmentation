import argparse
from visualization.utils.constants import (
    DEFAULT_CSV_FILES,
    DEFAULT_MODEL_NAMES,
    SIGNIFICANT_RESULTS,
    STYLE,
    CASES,
    INDIV_CASES,
    SOURCES_ENSEMBLE,
    SOURCES_INDIV,
)
from visualization.plotting.plot_metrics import plot_metrics
from visualization.plotting.prob_vs_uncertainty import plot_prob_vs_uncertainty
from visualization.plotting.important_cases import plot_important_cases
from visualization.plotting.uncertainty_maps import plot_uncertainty_maps

def main():
    parser = argparse.ArgumentParser(
        description="Visualization tools: metrics, prob_vs_unc, important, uncertainty_maps"
    )
    subparsers = parser.add_subparsers(dest="command")

    # -----------------------
    # Sub-command: metrics
    # -----------------------
    mp = subparsers.add_parser(
        "metrics", help="Plot HD95 / Dice / Sensitivity / Specificity barplots"
    )
    mp.add_argument("--csvs", "-c", nargs="+",
        default=DEFAULT_CSV_FILES,
        help="Metric CSV files (default: all ensemble + model CSVs)")
    mp.add_argument("--models", "-m", nargs="+",
        default=DEFAULT_MODEL_NAMES,
        help="Model names corresponding to each CSV")
    mp.add_argument("--metrics", "-k", nargs="+",
        default=["HD95 NCR", "HD95 ED", "HD95 ET"],
        help="Metric columns to plot")
    mp.add_argument("--palette", "-p", nargs="+",
        help="One color per metric (hex or name)")
    mp.add_argument("--sig", "-s",
        default=SIGNIFICANT_RESULTS,
        help="CSV of significant results for markers")
    mp.add_argument("--out", "-o",
        default="./figures/all_metrics.png",
        help="Where to save the figure")

    # -------------------------------
    # Sub-command: prob_vs_uncertainty
    # -------------------------------
    pp = subparsers.add_parser(
        "prob_vs_unc", help="Plot per-slice probability & uncertainty maps"
    )
    pp.add_argument("--data-dirs", "-d", nargs="+", required=True,
        help="Model output folders (must align with --models)")
    pp.add_argument("--models", "-m", nargs="+", required=True,
        help="Model names (same order as --data-dirs)")
    pp.add_argument("--patient", "-p", required=True,
        help="Patient ID (e.g. '00332')")
    pp.add_argument("--slice", "-s", type=int, default=None,
        help="Slice index (auto-select if omitted)")
    pp.add_argument("--sub-region", "-r", choices=["NCR","ED","ET"],
        default="ED", help="Tumor sub-region to plot")
    pp.add_argument("--out-dir", "-o", default=None,
        help="Where to save the probability/uncertainty figure")

    # -------------------------------
    # Sub-command: important cases
    # -------------------------------
    ip = subparsers.add_parser(
        "important", help="Plot selected important cases"
    )
    ip.add_argument("--which", "-w", choices=["ensemble","indiv"],
        default="ensemble",
        help="Use ensemble-defined cases or individual-defined cases")
    ip.add_argument("--out", "-o", default=None,
        help="Where to save the important-cases figure")

    # -------------------------------
    # Sub-command: uncertainty maps
    # -------------------------------
    uq = subparsers.add_parser(
        "uncertainty", help="Plot NCR/ED/ET uncertainty + pred + GT maps"
    )
    uq.add_argument("--data-dirs", "-d", nargs="+", required=True,
        help="Model output folders (must align with --models)")
    uq.add_argument("--models", "-m", nargs="+", required=True,
        help="Model names (same order as --data-dirs)")
    uq.add_argument("--patient", "-p", required=True,
        help="Patient ID (e.g. '00332')")
    uq.add_argument("--slice", "-s", type=int, default=None,
        help="Slice index (auto-select if omitted)")
    uq.add_argument("--out-dir", "-o", default=None,
        help="Where to save the uncertainty-maps figure")

    args = parser.parse_args()
    cmd = args.command or "metrics"

    if cmd == "metrics":
        palette = dict(zip(args.metrics, args.palette)) if args.palette else None
        plot_metrics(
            csv_files   = args.csvs,
            model_names = args.models,
            metrics     = args.metrics,
            palette     = palette,
            sig_csv     = args.sig,
            out_path    = args.out
        )

    elif cmd == "prob_vs_unc":
        out_path = plot_prob_vs_uncertainty(
            data_dirs   = args.data_dirs,
            model_names = args.models,
            patient_id  = args.patient,
            sub_region  = args.sub_region,
            slice_idx   = args.slice,
            out_dir     = args.out_dir
        )
        print(f"Saved probability/uncertainty maps to {out_path}")

    elif cmd == "important":
        if args.which == "ensemble":
            patient_list  = CASES
            model_sources = SOURCES_ENSEMBLE
        else:
            patient_list  = INDIV_CASES
            model_sources = SOURCES_INDIV

        out = plot_important_cases(
            patient_list=patient_list,
            model_sources=model_sources,
            out_path=args.out
        )
        print(f"Saved important cases figure to {out}")

    elif cmd == "uncertainty":
        out_path = plot_uncertainty_maps(
            data_dirs   = args.data_dirs,
            model_names = args.models,
            patient_id  = args.patient,
            slice_idx   = args.slice,
            out_dir     = args.out_dir
        )
        print(f"Saved uncertainty-maps figure to {out_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
