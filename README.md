# 💳 Credit Card Fraud Detection

This project uses supervised machine learning algorithms to detect fraudulent credit card transactions. It includes data cleaning, model training, hyperparameter tuning using GridSearchCV, evaluation, and saving the best-performing model.

---

## 📁 Project Structure

CreditCard-Fraud-Detection/
├── 01_data_cleaning.ipynb # 🧹 Cleans and preprocesses the dataset
├── 02_model_training.ipynb # 🤖 Trains and evaluates ML models
├── best_model.pkl # 🔒 Best-performing model (Random Forest) 
├── creditcard.csv # 📊 Dataset from Kaggle
├── README.md # 📘 Project overview and instructions


> ⚠️ **Note**: GitHub's upload limit is 25 MB, so large files are **not included** in this repository.
> - 📊 Dataset: [Download from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)


---

## 🚀 Workflow Summary

### 1. 🔄 Data Preprocessing
- Performed in `01_data_cleaning.ipynb`
- Loads raw dataset `creditcard.csv`
- Applies cleaning, scaling, and train-test split

### 2. 🤖 Model Training
- Done in `02_model_training.ipynb`
- Trains and compares:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Uses GridSearchCV for tuning
- Evaluates based on Accuracy, F1 Score, and AUC
- Saves the best model to `best_model.pkl`

---

## 🧪 How to Run the Project

### 1. Download Dataset
- Get the original dataset:  
  [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Place the file (`creditcard.csv`) in a `/data/` folder or update the path in the notebooks.

### 2. Run Data Preprocessing
```bash
Open and run 01_data_cleaning.ipynb
Open and run 02_model_training.ipynb
```
💼 Deployment Example
To use the trained model in any Python script:

import pickle

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Example usage:
# prediction = model.predict(scaled_input_data)

📎 Credits
Dataset: Kaggle – Credit Card Fraud Detection

Tools Used: Python, Jupyter Notebook, scikit-learn, pandas, numpy

👤 Author
Sounil Mandal – [GitHub](https://github.com/JOHNNY210702)
