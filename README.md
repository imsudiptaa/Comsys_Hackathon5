# 🚀 Comsys Hackathon 2025 
----------------------------

This repository contains solutions for two tasks:

- 🎯 **Task A**: Gender Classification 
- 🎯 **Task B**: Face Matching (Multi-Class Recognition with Distorted Inputs) 

All models are built using **PyTorch** and **Facenet-PyTorch**, trained and evaluated with robust metrics like Accuracy, Precision, Recall, and F1-Score. The codebase is fully modular, CLI-compatible, and includes pretrained weights and evaluation scripts.

---

## 📁 Repository Structure
**Comsys_Hackathon5**\
├──```Task_A_Gender_Classification```\
│ ├── src\
│ │ ├── train.py\
│ │ ├── eval.py\
│ │ └── test_task_a.py\
│ ├── saved_models\
│ │ └── gender_classifier_v1.pth\
│ ├── results\
│ │ ├── classification_report.txt\
│ │ ├── classification_report.json\
│ │ └── confusion_matrix.png\
│ └── model_diagram.png\
| └── README.md\
├──```Task_B_Face_Matching```\
│ ├── src\
│ │ ├── extract_embedding.py\
│ │ ├── match_distorted.py\
│ │ └── test_task_b.py\
│ ├── saved_models\
│ │ └── reference_embeddings.pt\
│ ├── results\
│ │ ├── val_match_results.json\
│ │ ├── val_match_results.csv\
│ │ ├── val_face_match_report.txt\
| └── README.md\
│ └── model_diagram.png\
└── README.md\
└── .gitattributes


 📌 **Task A** 
 
     📈Overview: Task A classifies gender (male / female) from face images using a fine-tuned ResNet-18 model.
     
     📊 Evaluation Metrics:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - Confusion Matrix
     
📌 **Task B** 

     📈Overview: Task B Verifies identity of distorted face images by matching with clean embeddings using FaceNet (InceptionResNetV1).
     
     📊 Evaluation Metrics: 
        - Top-1 Accuracy
        - Macro-averaged F1-Score
        - Precision
        - Recall

  ## 🧠 Model Architecture Diagrams
  Each task folder includes a `model_diagram.png` file to visualize the architecture.
     
  ## 📎 Dataset Download Link
  Since dataset files are large, you can download them from Google Drive:
  https://drive.google.com/drive/folders/1lRW5ipuIjo7z2LDiLfumhFIbOies2_7A?usp=drive_link

  <br>

  Contact
  --------
  📧 Email : imsudiptaa@gmail.com

