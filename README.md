# 🎧 Spotify Hit Predictor

Predict whether a song will be a hit or not using audio features like **danceability**, **energy**, **tempo**, and **valence**.

This machine learning project uses PyCaret to build a classification model that predicts the likelihood of a track becoming popular based solely on its musical attributes — no artist names or Spotify IDs involved.

---

## 🚀 Project Overview

- **Objective**: Predict song popularity (hit or flop) based on audio features.
- **Dataset**: [Spotify Hit Predictor Dataset (1960–2019)](https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset)
- **Modeling Framework**: [PyCaret](https://pycaret.gitbook.io/docs/)
- **Algorithm Used**: CatBoost Classifier (best-performing model)

---

## 📁 Files Included

- `spotify_hit_predictor_model.pkl`: Final trained model file
- `notebook.ipynb`: All data processing, training, and evaluation steps
- `predicted_hits.csv` *(optional)*: Predictions made on test data

---

## 🎯 Features Used

Only real audio features were used (no IDs or artist names):

- `danceability`
- `energy`
- `tempo`
- `valence` *(mood)*
- `acousticness`
- `instrumentalness`
- `speechiness`
- `loudness`
- `liveness`
- `duration_ms`

---

## 💡 Model Performance

| Metric      | Score   |
|-------------|---------|
| Accuracy    | 89.7%   |
| AUC         | 0.98    |
| Precision   | 98.9%   |
| Recall      | 80.3%   |
| F1 Score    | 88.6%   |

---

## 🔄 How to Use the Model

1. Clone this repo or download the `.pkl` model.
2. Install PyCaret:

```bash
pip install pycaret

**Use in Python:**
from pycaret.classification import load_model, predict_model
import pandas as pd

model = load_model('spotify_hit_predictor_model')

new_data = pd.DataFrame([{
    'danceability': 0.82,
    'energy': 0.71,
    'tempo': 115,
    'valence': 0.91,
    'acousticness': 0.14,
    'instrumentalness': 0.0,
    'speechiness': 0.04,
    'loudness': -5.2,
    'liveness': 0.09,
    'duration_ms': 200000
}])

predict_model(model, data=new_data)
