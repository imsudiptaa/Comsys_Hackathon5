import os
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image)

def extract_embeddings_and_evaluate(train_dir, output_path, results_dir, device):
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Extract reference embeddings from clean images
    print("\n Extracting reference embeddings from clean images...")
    embeddings_dict = {}
    person_folders = sorted(os.listdir(train_dir))
    person_to_index = {person: idx for idx, person in enumerate(person_folders)}

    for person in tqdm(person_folders, desc="Reference (clean)"):
        person_path = os.path.join(train_dir, person)
        clean_images = [f for f in os.listdir(person_path)
                        if os.path.isfile(os.path.join(person_path, f)) and not f.startswith(".")]

        if not clean_images:
            print(f" No clean image found for {person}")
            continue

        clean_image_path = os.path.join(person_path, clean_images[0])
        img_tensor = load_image(clean_image_path).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(img_tensor).squeeze(0)

        embeddings_dict[person] = embedding

    #  Evaluate using distorted images
    print("\nMatching distorted images against clean reference embeddings...")
    all_labels, all_preds = [], []

    for person in tqdm(person_folders, desc="Distorted Matching"):
        person_path = os.path.join(train_dir, person, "distortion")

        if not os.path.exists(person_path):
            print(f" No distortion folder for {person}")
            continue

        distorted_images = [f for f in os.listdir(person_path)
                            if os.path.isfile(os.path.join(person_path, f)) and not f.startswith(".")]

        for img_file in distorted_images:
            img_path = os.path.join(person_path, img_file)
            img_tensor = load_image(img_path).unsqueeze(0).to(device)

            with torch.no_grad():
                distorted_embedding = model(img_tensor).squeeze(0)

            # Match with reference embeddings
            similarities = {
                ref_person: torch.nn.functional.cosine_similarity(
                    distorted_embedding.unsqueeze(0), ref_emb.unsqueeze(0)
                ).item()
                for ref_person, ref_emb in embeddings_dict.items()
            }

            predicted_person = max(similarities, key=similarities.get)
            all_labels.append(person_to_index[person])
            all_preds.append(person_to_index[predicted_person])

    #Evaluation metrics
    acc = accuracy_score(all_labels, all_preds) * 100
    prec = precision_score(all_labels, all_preds, average='macro') * 100
    rec = recall_score(all_labels, all_preds, average='macro') * 100
    f1 = f1_score(all_labels, all_preds, average='macro') * 100

    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embeddings_dict, output_path)

    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "train_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Training Top-1 Accuracy: {acc:.2f}%\n")
        f.write(f"Training Precision: {prec:.2f}%\n")
        f.write(f"Training Recall: {rec:.2f}%\n")
        f.write(f"Training Macro F1 Score: {f1:.2f}%\n")

    print(f"\nEvaluation Results:")
    print(f"Top-1 Accuracy: {acc:.2f}%")
    print(f"Precision: {prec:.2f}%")
    print(f"Recall: {rec:.2f}%")
    print(f"Macro F1 Score: {f1:.2f}%")
    print(f"\n Metrics saved to: {metrics_path}")
    print(f" Reference embeddings saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate Reference Embeddings (Task B)")
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training folder')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save reference_embeddings.pt')
    parser.add_argument('--results_dir', type=str, default="results", help='Path to save evaluation metrics')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extract_embeddings_and_evaluate(args.train_dir, args.output_path, args.results_dir, device)

if __name__ == "__main__":
    main()
