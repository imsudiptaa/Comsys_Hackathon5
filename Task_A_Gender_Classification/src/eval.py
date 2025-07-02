import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import argparse

def load_model(model_path, device):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def evaluate(model, val_loader, class_names, device, results_dir):
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Evaluation Metrics
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='macro') * 100
    rec = recall_score(y_true, y_pred, average='macro') * 100
    f1 = f1_score(y_true, y_pred, average='macro') * 100

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"\nSummary:\nAccuracy: {acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1 Score: {f1:.2f}%")

    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "classification_report.json"), "w") as f_json:
        json.dump(classification_report(y_true, y_pred, target_names=class_names, output_dict=True), f_json, indent=4)

    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f_txt:
        f_txt.write(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

def main(val_path, model_path, results_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    val_dataset = datasets.ImageFolder(val_path, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    class_names = val_dataset.classes

    model = load_model(model_path, device)
    evaluate(model, val_loader, class_names, device, results_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Gender Classification Model (Task A)")
    parser.add_argument('--val_path', type=str, required=True, help='Path to validation data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to save evaluation reports')

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print("Model file not found!")
    elif not os.path.exists(args.val_path):
        print("Validation folder not found!")
    else:
        main(args.val_path, args.model_path, args.results_dir)
