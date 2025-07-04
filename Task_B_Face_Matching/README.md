🚀 Comsys Hackathon 2025
-------------------------

# Task B - Face matching (Multi-Class Recognition with Distorted Inputs) 

## Overview
This system performs identity recognition by matching distorted facial images to their corresponding clean reference images.
It uses FaceNet (InceptionResnetV1) to generate facial embeddings and compares them using cosine similarity.

## Directory Structure
![Screenshot 2025-07-03 171019](https://github.com/user-attachments/assets/d5ee13c4-9d41-4e10-8f76-9d3e0e647c02)



- `extract_embedding.py`: Extracts clean reference embeddings from training set
- `match_distorted.py`: Matches distorted images in validation set to reference embeddings
- `test_task_b.py`: Final Test script for Task B evaluation
- `src/`: contains all source codes 
- `saved_models/`: Stores extracted clean embeddings from training set
- `results/`: Contains similarity scores, classification reports, and evaluation metrics
- `model_diagram.png`: Architecture diagram


## Requirements
- Python 3.8+
- torch
- torchvision
- facenet-pytorch
- scikit-learn
- tqdm
- matplotlib
- Pillow

## Installation 
To run this task, you need to install the following packages
```bash
pip install torch torchvision facenet-pytorch scikit-learn tqdm matplotlib Pillow
```

## How to Use

### 1. Extract Clean Image Embeddings (from training set)
```bash
python /content/drive/MyDrive/Comsys_Hackathon5/Task_B_Face_Matching/src/extract_embedding.py \
--train_dir "/content/drive/MyDrive/Comys_Hackathon5/Task_B/train" \
--output_path "/content/drive/MyDrive/Comsys_Hackathon5/Task_B_Face_Matching/saved_models/reference_embeddings.pt"
```
### 2.  Evaluate Matching on Validation Set
```bash
python /content/drive/MyDrive/Comsys_Hackathon5/Task_B_Face_Matching/src/match_distorted.py
```
### 3. Final Test Script
```bash
python /content/drive/MyDrive/Comsys_Hackathon5/Task_B_Face_Matching/src/test_task_b.py \
--val_path "/content/drive/MyDrive/Comys_Hackathon5/Task_B/val" \
--results_dir "/content/drive/MyDrive/Comsys_Hackathon5/Task_B_Face_Matching/results" \
--threshold 0.48

```

## Outputs
- `val_match_results.csv`: Shows similarity score and label for each distorted image
- `val_match_results.json`: Machine-readable version of results
- `val_face_match_report.txt`: Shows Top-1 Accuracy, Macro-averaged F1-Score, Accuracy, Precision, Recall

## Model Architecture
![model_diagram](https://github.com/user-attachments/assets/04e25478-5171-44e0-b58c-b8a4c6b66bad)
