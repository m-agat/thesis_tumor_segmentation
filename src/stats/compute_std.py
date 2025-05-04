import argparse
from stats.data import load_model_performances
from stats.metrics import add_region_averages, summarize_metrics

def main(perf_files: str, metrics: str, regions: str):
    # Parse comma-separated args
    paths = {}
    for entry in perf_files.split(","):
        try:
            name, csv_path = entry.split(":", 1)
        except ValueError:
            raise ValueError(
                f"Invalid --perf-files entry: '{entry}'. "
                "Must be in the form Name:Path"
            )
        paths[name.strip()] = csv_path.strip()
    metric_list = [m.strip() for m in metrics.split(",")]        
    region_list = [r.strip() for r in regions.split(",")]        
    
    # Compute standard deviation
    dfs = load_model_performances(paths)
    add_region_averages(dfs, metric_list, region_list)
    summary = summarize_metrics(dfs, metric_list, region_list)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        prog="compute-std",
        description="Compute mean & std of metrics per model & region"
    )
    p.add_argument(
        "--perf-files", required=True,
        help="Comma-separated list of model_name:csv_path"
    )
    p.add_argument(
        "--metrics", required=True,
        help="Comma-separated base metric names (e.g. Dice,Sensitivity,HD95)"
    )
    p.add_argument(
        "--regions", required=True,
        help="Comma-separated region suffixes (e.g. NCR,ED,ET)"
    )
    args = p.parse_args()
    main(args.perf_files, args.metrics, args.regions)
