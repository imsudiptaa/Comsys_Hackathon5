import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_data_loaders(train_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def evaluate_on_training_data(model, train_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='macro') * 100
    rec = recall_score(y_true, y_pred, average='macro') * 100
    f1 = f1_score(y_true, y_pred, average='macro') * 100
    return acc, prec, rec, f1

def train_model(train_loader,  model, criterion, optimizer, device, num_epochs,results_dir):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Save Training Metrics After Training
    acc, prec, rec, f1 = evaluate_on_training_data(model, train_loader, device)
    train_results_dir = results_dir
    os.makedirs(train_results_dir, exist_ok=True)

    metrics_path = os.path.join(train_results_dir, "train_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Training Accuracy: {:.2f}%\n".format(acc))
        f.write("Training Precision: {:.2f}%\n".format(prec))
        f.write("Training Recall: {:.2f}%\n".format(rec))
        f.write("Training F1 Score: {:.2f}%\n".format(f1))

  
    print("\nTraining Metrics:")
    print("Training Accuracy: {:.2f}%".format(acc))
    print("Training Precision: {:.2f}%".format(prec))
    print("Training Recall: {:.2f}%".format(rec))
    print("Training F1 Score: {:.2f}%".format(f1))
    print("\nTraining metrics saved to:", metrics_path)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_data_loaders(args.train_dir, args.batch_size)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_model(train_loader, model, criterion, optimizer, device, args.epochs,args.results_dir)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"\n Model saved to: {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a gender classification model (Task A)")
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save trained model')
    parser.add_argument('--results_dir', type=str, default="results", help='Directory to save training metrics')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    main(args)
