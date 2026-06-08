# 🏥 Health Insurance Fraud Detection System

A Machine Learning-based web application that detects fraudulent health insurance claims using multiple algorithms and real-time prediction.

---

## 🚀 Project Overview

Insurance fraud is a major issue in the healthcare industry, leading to financial losses and inefficiencies.  
This project aims to automatically detect fraudulent claims using Machine Learning models trained on a realistic dataset.

The system allows users to input claim details and get predictions along with fraud probability.

---

## 🎯 Features

- 🔍 Fraud detection using ML models
- 📊 Probability-based prediction output
- ⚖️ Comparison of 3 algorithms:
  - Random Forest 🌲
  - Logistic Regression 📊
  - Decision Tree 🌳
- 🧠 Context-aware predictions (based on diagnosis & medical conditions)
- 🌐 Full-stack web application (React + Flask)
- 📈 Large dataset (1600+ records) for better accuracy

---

## 🧠 Machine Learning Models

### 🌲 Random Forest
- Ensemble learning method
- Combines multiple decision trees
- High accuracy and reduced overfitting

### 🌳 Decision Tree
- Rule-based model
- Easy to interpret
- Fast but prone to overfitting

### 📊 Logistic Regression
- Probability-based model
- Uses sigmoid function
- Good for understanding feature impact

---

## ⚙️ Tech Stack

### Frontend
- React.js
- Axios
- CSS

### Backend
- Flask (Python)
- Pandas, NumPy
- Scikit-learn

---

## 📂 Project Structure
health-fraud-detection/
│
├── backend/
│ ├── app.py
│ ├── model.py
│ ├── fraud_dataset.csv
│ ├── rf_model.pkl
│ ├── lr_model.pkl
│ ├── dt_model.pkl
│ ├── scaler.pkl
│ ├── imputer.pkl
│ └── columns.pkl
│
├── frontend/
│ ├── src/
│ │ ├── App.js
│ │ └── App.css
│ └── package.json
│
└── README.md
---

### 🔄 Workflow

1. User enters claim details in the UI  
2. Data is sent to Flask backend  
3. Backend preprocesses data:
   - Encoding
   - Imputation
   - Scaling  
4. Models predict fraud probability  
5. Results from all 3 models are returned  
6. Frontend displays comparison  

---

## 🧪 Sample Input

| Feature | Example |
|--------|--------|
| Claim Amount | 3500000 |
| Age | 30 |
| Stay Days | 1 |
| Previous Claims | 8 |
| Deductible | 500 |
| CoPay | 200 |
| Late | Yes |
| Diagnosis | Fever |

---

## 📊 Sample Output

```

Random Forest → Fraud (82%)
Logistic Regression → Genuine (35%)
Decision Tree → Fraud (90%)

Best Model → Decision Tree
```

---
### ▶️ How to Run
## 🔹 Backend Setup

```bash
cd backend
pip install -r requirements.txt
python model.py
python app.py
```
## Backend runs on
```bash
http://127.0.0.1:5000
```
### 🔹 Frontend Setup
```bash
cd frontend
npm install
npm start
```
## Frontend runs on
```bash
http://localhost:3000
```
### 📌 Key Highlights
💡 Real-world dataset with medical conditions & accidents
⚙️ End-to-end ML pipeline (Preprocessing → Prediction)
📊 Multi-model comparison for better reliability
🧠 Intelligent fraud detection logic
📈 Future Improvements
Use real-world insurance datasets
Deploy on cloud (AWS / Render)
Add authentication system
Improve UI/UX (charts, dashboards)
Use deep learning models
### 👩‍💻 Contributors
Qudsia Fatima
M. Kalyani
Sarah Hasan
### 🎓 Conclusion

This project demonstrates how Machine Learning can be applied to solve real-world problems like healthcare fraud detection.
It improves efficiency, reduces manual effort, and enables faster and more reliable decision-making.
⭐ If you found this useful, consider giving it a star!