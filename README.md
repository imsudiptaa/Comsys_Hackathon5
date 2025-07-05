# 🚀 Comsys Hackathon 2025 
----------------------------
## 🌟 Team Name: Drishti
### Team members : Sudipta Mandal (Team Leader), Khushi Kumari Gupta & Ajiti Kumari Shaw
-----------------------------------------------------------------------------------------

This repository contains solutions for two tasks:

- 🎯 **Task A**: Gender Classification 
- 🎯 **Task B**: Face Matching (Multi-Class Recognition with Distorted Inputs)
- Each Task folders `Task_A_Gender_Classification` & `Task_B_Face_Matching` consist separate `README.md` files containing overview, directory structure, requirements, Installation guide, Train, Evaluation, Final Test script, Outputs & Model Architecture diagram.

All models are built using **PyTorch** and **Facenet-PyTorch**, trained and evaluated with robust metrics like Accuracy, Precision, Recall, and F1-Score. The codebase is fully modular, CLI-compatible, and includes pretrained weights and evaluation scripts.

---

## 📁 Repository Structure
**Comsys_Hackathon5**\
├──```Task_A_Gender_Classification``` (⚛ Contains solutions of Task A) <br>
│├── src (⚛ Contains all source codes of Task A) \
│ │ ├── train.py\
│ │ ├── eval.py\
│ │ └── test_task_a.py\
│ ├── saved_models (⚛ Contains model weights of Task A )\
│ │ └── gender_classifier_v1.pth\
│ ├── results (⚛ Contains Evaluation metrics of Task A )\
│ │ ├── classification_report.txt\
│ │ ├── classification_report.json\
│ │ └── confusion_matrix.png\
│ └── model_diagram.png (⚛ Contains model architecture of Task A )\
| └── README.md (⚛ Contains detailed `Readme.md` file of Task A )\
├──```Task_B_Face_Matching``` (💠 Contains solutions of Task B ) <br>
│ ├── src (💠 Contains source codes of Task B )\
│ │ ├── extract_embedding.py\
│ │ ├── match_distorted.py\
│ │ └── test_task_b.py\
│ ├── saved_models (💠 Contains model weights of Task B )\
│ │ └── reference_embeddings.pt\
│ ├── results (💠 Contains evaluation metrics of Task B )\
│ │ ├── val_match_results.json\
│ │ ├── val_match_results.csv\
│ │ ├── val_face_match_report.txt\
| └── README.md (💠 Contains detailed `Readme.md` file of Task B )\
│ └── model_diagram.png (💠 Contains model architecture of Task B )\
└── README.md (🏀`Readme.md` file of root repo)\
└── .gitattributes
└── Technical Summmary.png (🏀 Contains Technical Summary of the tasks)


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

