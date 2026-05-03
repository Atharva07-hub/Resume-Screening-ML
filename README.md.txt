# AI Resume Screening System

## Overview
This project builds a machine learning model to classify resumes as relevant or not relevant for Information Technology roles using Natural Language Processing techniques.

## Approach
- Text preprocessing (cleaning + stopword removal)
- TF-IDF vectorization
- Logistic Regression with class balancing
- Threshold tuning to optimize precision and recall

## Results
- Accuracy: ~96–97%
- Precision (IT class): ~0.68
- Recall (IT class): ~0.81
- F1 Score: ~0.74

## Key Insight
The dataset was imbalanced, which caused biased predictions. This was handled using class weighting and further improved by tuning the classification threshold.

## Tech Stack
- Python
- Pandas
- Scikit-learn
- NLTK

## Dataset
Resume dataset containing multiple job categories (Kaggle-based)

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook
```