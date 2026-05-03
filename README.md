# AI Resume Screening System (IT Role Classification)

## 🧠 Overview
This project builds a machine learning model to classify resumes as relevant or not relevant for Information Technology (IT) roles using Natural Language Processing (NLP) techniques. The system automates resume screening, reducing manual effort in recruitment processes.

---

## 🧩 Problem Statement
Manual resume screening is time-consuming and inefficient, especially when recruiters receive hundreds of applications. Identifying relevant candidates quickly becomes challenging. This project aims to automate the screening process using machine learning to classify resumes based on their relevance to IT roles.

---

## 🔄 Workflow
1. Input resume text  
2. Text preprocessing (cleaning, removing stopwords)  
3. Feature extraction using TF-IDF  
4. Train classification model  
5. Handle class imbalance using class weights  
6. Tune decision threshold for better precision-recall balance  
7. Predict whether a resume is relevant or not  

---

## ⚙️ Approach
- Text preprocessing using NLTK  
- TF-IDF vectorization (max_features=3000)  
- Logistic Regression model  
- Class imbalance handled using `class_weight='balanced'`  
- Threshold tuning to improve model performance  

---

## 📊 Results
- Accuracy: ~96%  
- Precision (IT class): ~0.68  
- Recall (IT class): ~0.81  
- F1 Score: ~0.74  

---

## ⚖️ Model Optimization
The initial model showed high accuracy but poor recall for IT resumes due to class imbalance. This issue was addressed using:
- Class weighting to balance minority class  
- Threshold tuning (final threshold = 0.7) to improve precision-recall trade-off  

---

## 🛠️ Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- NLTK  

---

## 📂 Dataset
Resume dataset containing multiple job categories (Kaggle-based).  
For this project, resumes labeled as **INFORMATION-TECHNOLOGY** were considered relevant.

---

## ▶️ How to Run
```bash
pip install -r requirements.txt
jupyter notebook
