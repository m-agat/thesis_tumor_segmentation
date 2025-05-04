import argparse
from box import plot_box
from bar import plot_grouped_bar
from violin import plot_violin
from radar import plot_radar
from visualization.utils.io import load_json

def main():
    p = argparse.ArgumentParser(
        description="Dice-metric visualizations: bar, box, violin, radar"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # grouped-bar
    b = sub.add_parser("bar")
    b.add_argument("--base-path", "-b", required=True)
    b.add_argument("--models", "-m", nargs="+", required=True)
    b.add_argument("--metrics", "-k", nargs="+", required=True)
    b.add_argument("--json", "-j", required=True, help="average JSON file name")
    b.add_argument("--csv", "-c", required=True, help="per-patient CSV file name")
    b.add_argument("--out", "-o", required=True)
    b.set_defaults(func=lambda args: plot_grouped_bar(
        base_path   = args.base_path,
        group_names = dict(zip(args.models, args.models)),
        metrics     = args.metrics,
        region_colors={m: None for m in args.metrics},
        json_file   = args.json,
        csv_file    = args.csv,
        out_path    = args.out,
    ))

    # box
    x = sub.add_parser("box")
    x.add_argument("--base-path", "-b", required=True)
    x.add_argument("--models", "-m", nargs="+", required=True)
    x.add_argument("--metrics", "-k", nargs="+", required=True)
    x.add_argument("--pattern", "-p", required=True,
                   help="filename pattern, e.g. '{key}_patient_metrics_test.csv'")
    x.add_argument("--out-combined",   required=True)
    x.add_argument("--out-individual", required=True)
    x.set_defaults(func=lambda args: plot_box(
        base_path      = args.base_path,
        group_names    = dict(zip(args.models, args.models)),
        metrics        = args.metrics,
        plot_type      = "box",
        file_pattern   = args.pattern,
        out_combined   = args.out_combined,
        out_individual = args.out_individual,
    ))

    # violin
    v = sub.add_parser("violin")
    v.add_argument("--base-path", "-b", required=True)
    v.add_argument("--models", "-m", nargs="+", required=True)
    v.add_argument("--metrics", "-k", nargs="+", required=True)
    v.add_argument("--pattern", "-p", required=True)
    v.add_argument("--out-combined",   required=True)
    v.add_argument("--out-individual", required=True)
    v.set_defaults(func=lambda args: plot_violin(
        base_path      = args.base_path,
        group_names    = dict(zip(args.models, args.models)),
        metrics        = args.metrics,
        file_pattern   = args.pattern,
        out_combined   = args.out_combined,
        out_individual = args.out_individual,
    ))

    # radar
    r = sub.add_parser("radar")
    r.add_argument("--stats-json", "-s", nargs="+", required=True,
                   help="one JSON per model, in same order as --models")
    r.add_argument("--models", "-m", nargs="+", required=True)
    r.add_argument("--metrics", "-k", nargs="+", required=True)
    r.add_argument("--out", "-o", required=True)
    def _radar(args):
        stats = {
            model: load_json(path)
            for model, path in zip(args.models, args.stats_json)
        }
        plot_radar(
            stats       = stats,
            model_names = args.models,
            metrics     = args.metrics,
            colors      = [None]*len(args.models),
            out_path    = args.out,
        )
    r.set_defaults(func=_radar)

    args = p.parse_args()
    args.func(args)

if __name__=="__main__":
    main()

