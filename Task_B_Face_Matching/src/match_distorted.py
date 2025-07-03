import os
import torch
import json
import csv
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from facenet_pytorch import InceptionResnetV1

def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image)

def cosine_similarity(emb1, emb2):
    return torch.nn.functional.cosine_similarity(emb1, emb2)

def evaluate_on_val(val_dir, results_dir, threshold=0.48):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    results = []
    y_true = []
    y_pred = []

    print("Matching distorted images in VAL set against corresponding clean images...")
    person_folders = os.listdir(val_dir)

    for person in tqdm(person_folders):
        person_path = os.path.join(val_dir, person)
        if not os.path.isdir(person_path):
            continue

        # Loading all clean images for this person (excluding the distortion folder)
        clean_embeddings = []
        for file in os.listdir(person_path):
            file_path = os.path.join(person_path, file)
            if os.path.isfile(file_path) and not file.startswith("."):
                img = load_image(file_path).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(img)
                clean_embeddings.append(emb.squeeze(0))

        if not clean_embeddings:
            continue

        # Average the clean embeddings 
        ref_embedding = torch.stack(clean_embeddings).mean(dim=0).unsqueeze(0)

        # Processing distorted images
        distortion_path = os.path.join(person_path, "distortion")
        if not os.path.exists(distortion_path):
            continue

        for distorted_file in os.listdir(distortion_path):
            distorted_path = os.path.join(distortion_path, distorted_file)
            if os.path.isfile(distorted_path):
                img = load_image(distorted_path).unsqueeze(0).to(device)
                with torch.no_grad():
                    distorted_emb = model(img)

                score = cosine_similarity(distorted_emb, ref_embedding).item()
                label = 1 if score >= threshold else 0

                y_true.append(1)  
                y_pred.append(label)

                results.append({
                    "image": distorted_file,
                    "identity": person,
                    "similarity_score": round(score, 4),
                    "label": label
                })

    # Saving results
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "val_match_results.json"), "w") as jf:
        json.dump(results, jf, indent=2)

    with open(os.path.join(results_dir, "val_match_results.csv"), "w", newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Evaluation
    correct_matches = sum(y_pred)
    total = len(y_true)
    top1_acc = correct_matches / total * 100
    macro_f1 = f1_score(y_true, y_pred, average='macro') * 100

    print("\nEvaluation Metrics:")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Macro-averaged F1-Score: {macro_f1:.2f}%")

    with open(os.path.join(results_dir, "val_face_match_report.txt"), "w") as f:
        f.write(f"Top-1 Accuracy: {top1_acc:.2f}%\n")
        f.write(f"Macro-averaged F1-Score: {macro_f1:.2f}%\n")
        f.write("\nDetailed Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=["Mismatch (0)", "Match (1)"]))

    print("\nSaved report and results to:", results_dir)

    return y_pred 
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, zero_division=0) * 100
    recall = recall_score(y_true, y_pred, zero_division=0) * 100

    print("\nAdditional Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")

    with open(os.path.join(results_dir, "val_face_match_report.txt"), "a") as f:
        f.write(f"\nAdditional Metrics:\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Precision: {precision:.2f}%\n")
        f.write(f"Recall: {recall:.2f}%\n")
