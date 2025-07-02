🚀 Comsys Hackathon 2025
-------------------------

# Task A - Gender Classification 

## Overview
This model classifies images into two genders: Male or Female. It uses a fine-tuned ResNet-18 architecture with PyTorch.

## Directory Structure
![Screenshot 2025-07-02 210142](https://github.com/user-attachments/assets/37b00af5-4163-423b-b953-ad8cb04ac845)

- `train.py`: Script to train the model
- `eval.py`: Evaluation script
- `test_task_a.py`: Test script for final evaluation
- `src/`: contains all source codes 
- `saved_models/`: Contains pretrained model weights
- `results/`: Stores classification reports and confusion matrix
- `model_diagram.png`: Architecture diagram


## Requirements
- Python 3.8+
- PyTorch
- torchvision
- sklearn
- matplotlib

## How to Use

### 1. Train
```bash
!python /content/drive/MyDrive/Comsys_Hackathon5/Task_A_Gender_Classification/src/train.py 
```
### 2. Evaluation
```bash
!python /content/drive/MyDrive/Comsys_Hackathon5/Task_A_Gender_Classification/src/eval.py \
--val_path "/content/drive/MyDrive/Comys_Hackathon5/Task_A/val" \
--model_path "/content/drive/MyDrive/Comsys_Hackathon5/Task_A_Gender_Classification/saved_models/gender_classifier_v1.pth"
```
### 3. Final Test Script
```bash
!python /content/drive/MyDrive/Comsys_Hackathon5/Task_A_Gender_Classification/src/test_task_a.py \
--val_path "/content/drive/MyDrive/Comys_Hackathon5/Task_A/val" \
--model_path "/content/drive/MyDrive/Comsys_Hackathon5/Task_A_Gender_Classification/saved_models/gender_classifier_v1.pth"
```

## Outputs
- `classification_report.txt`: Shows Accuracy, Precision, Recall, F1-Score.
- `confusion_matrix.png`: Shows Visual Matrix.

