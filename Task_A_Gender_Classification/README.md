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
python src/train.py --val_path path/to/train --model_path saved_models/gender_classifier_v1.pth
```
### 2. Evaluation
```bash
python src/eval.py --val_path path/to/val --model_path saved_models/gender_classifier_v1.pth
```
### 3. Final Test Script
```bash
python src/test_task_a.py --val_path path/to/val --model_path saved_models/gender_classifier_v1.pth
```

