import argparse
import os 
from model_runner import load_predict_models, create_data_loader, run_predictions

def parse_args():
    p = argparse.ArgumentParser("OOD Prediction")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--methods", nargs="+",
                   choices=["simple","perf","tta","ttd","hybrid"],
                   default=["simple"])
    p.add_argument("--n-iter", type=int, default=10)
    p.add_argument("--patient_id", type=str, default="UNKNOWN")

    return p.parse_args()

def get_preprocessed_files(directory):
    """
    Get list of preprocessed files from a directory.
    Only returns files that have 'preproc_' in their name.
    
    Args:
        directory (str): Path to directory containing preprocessed files
        
    Returns:
        list: List of full paths to preprocessed files
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
        
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and 'preproc_' in f]

def main():
    args = parse_args()
    models = load_predict_models()
    loader = create_data_loader(get_preprocessed_files(args.input_dir), args.patient_id)
    run_predictions(
        loader, 
        models,
        methods=args.methods,
        out_base=args.output_dir,
        n_iter=args.n_iter
    )

if __name__=="__main__":
    main()