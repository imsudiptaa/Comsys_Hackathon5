import os
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image)

def extract_reference_embeddings(train_dir, output_path, device):
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    embeddings_dict = {}

    person_folders = os.listdir(train_dir)
    for person in tqdm(person_folders, desc="Extracting clean image embeddings"):
        person_path = os.path.join(train_dir, person)
        distortion_path = os.path.join(person_path, 'distortion')

        clean_images = [f for f in os.listdir(person_path)
                        if os.path.isfile(os.path.join(person_path, f)) and not f.startswith(".")]

        if not clean_images:
            print(f"Warning: No clean image found for {person}")
            continue

        clean_image_path = os.path.join(person_path, clean_images[0])
        img_tensor = load_image(clean_image_path).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(img_tensor).squeeze(0)

        embeddings_dict[person] = embedding

    torch.save(embeddings_dict, output_path)
    print(f"\n Saved clean image embeddings to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract reference face embeddings (Task B)")
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the training folder with identity subfolders')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save embeddings (reference_embeddings.pt)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.train_dir):
        raise FileNotFoundError("Training directory not found.")
    
    extract_reference_embeddings(args.train_dir, args.output_path, device)

if __name__ == "__main__":
    main()
