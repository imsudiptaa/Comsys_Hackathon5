import argparse
import os
from eval import evaluate_on_val

def main(val_path, results_dir, threshold):
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation path not found: {val_path}")
    os.makedirs(results_dir, exist_ok=True)

    evaluate_on_val(val_path, results_dir, threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Face Matching Model (Task B)")
    parser.add_argument('--val_path', type=str, required=True, help='Path to validation data')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--threshold', type=float, default=0.48, help='Cosine similarity threshold (default=0.48)')
    args = parser.parse_args()

    main(args.val_path, args.results_dir, args.threshold)
