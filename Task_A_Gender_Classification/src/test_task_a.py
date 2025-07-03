import argparse
from eval import main as evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run gender classification test script")
    parser.add_argument("--val_path", type=str, required=True,
                        help="Path to validation data (with 'male/' and 'female/' subfolders)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model (.pth file)")
    parser.add_argument('--results_dir', type=str, required=True, 
                        help='Directory to save evaluation reports')

    args = parser.parse_args()
    evaluate_model(args.val_path, args.model_path, args.results_dir)
