import argparse
from prediction_visualizer import PredictionVisualizer
from uncertainty_visualizer import UncertaintyVisualizer

def main():
    parser = argparse.ArgumentParser(
        description="Visualize model predictions or uncertainties for given patients."
    )
    parser.add_argument(
        "mode",
        choices=["pred", "unc"],
        help="Visualization mode: 'pred' for comparisons, 'unc' for uncertainty maps",
    )
    parser.add_argument(
        "model_type",
        choices=["ensemble", "indiv"],
        help="Type of model to visualize",
    )
    parser.add_argument(
        "--patients",
        nargs='+',
        default=["VIGO_01", "VIGO_03"],
        help="List of patient IDs",
    )
    parser.add_argument(
        "--models",
        nargs='+',
        help="List of model names to use (only used in 'unc' mode)",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the output figure",
    )

    args = parser.parse_args()

    # Instantiate appropriate visualizer
    if args.mode == "pred":
        viz = PredictionVisualizer(model_type=args.model_type)
        viz.create_comparison_figure(
            patients=args.patients,
            models=args.models,
            save_path=args.save_path or "./Figures/case_comparison.png",
        )
    else:
        viz = UncertaintyVisualizer(model_type=args.model_type)
        # default single-models if none provided
        single_models = args.models or ["ttd", "tta", "hybrid"]
        viz.create_uncertainty_figure(
            patients=args.patients,
            models=single_models,
            save_path=args.save_path or "./Figures/uncertainty_maps.png",
        )


if __name__ == "__main__":
    main()
