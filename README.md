# 🏥 Health Insurance Fraud Detection System

A Machine Learning-based web application that predicts whether a health insurance claim is **Genuine** or **Fraudulent** based on claim details and medical conditions.

---

## 📌 Overview

Health insurance fraud is a major challenge in the healthcare industry, leading to financial losses and inefficiencies. This project aims to automate fraud detection using Machine Learning.

The system analyzes user input such as claim amount, patient details, claim history, and diagnosis to predict fraud probability and classification.

---

## 🚀 Features

- 🔍 Detects fraudulent insurance claims
- 📊 Displays fraud probability score
- 🧠 Uses Machine Learning (Random Forest)
- 🏥 Includes medical condition-based analysis
- ⚡ Real-time prediction through web interface
- 🎯 Simple and user-friendly UI

---

## 🧠 Technologies Used

- **Frontend:** React.js  
- **Backend:** Flask (Python)  
- **Machine Learning:** Scikit-learn (Random Forest)  
- **Data Processing:** Pandas, NumPy  

---

## 📊 Input Fields

The system takes the following inputs:

- Claim Amount (₹)
- Patient Age
- Length of Stay (Days)
- Number of Previous Claims
- Deductible Amount
- CoPay Amount
- Claim Submitted Late (Yes/No)
- Diagnosis (Fever, Cold, Accident, Cancer, etc.)

---

## 🎯 Output

- **Prediction:**  
  - Genuine Claim  
  - Fraudulent Claim  

- **Fraud Probability (%):**  
  Indicates the likelihood of fraud

---

## 🧪 Example Use Cases

| Scenario | Result |
|--------|--------|
| Low claim + minor illness | ✅ Genuine |
| High claim + serious condition | ✅ Genuine |
| High claim + minor illness | ❌ Fraud |
| Late claim + many previous claims | ❌ Fraud |

---

## ⚙️ How to Run the Project

### 🔹 Backend Setup

```bash
cd backend
python app.py

Backend runs on:

http://127.0.0.1:5000
🔹 Frontend Setup
cd frontend
npm install
npm start

Frontend runs on:

http://localhost:3000
🔄 Workflow
User enters claim details in the web interface
Data is sent to Flask backend
Backend preprocesses data (encoding, scaling)
Machine learning model predicts fraud
Result + probability is sent back to frontend
Output is displayed to the user


📌 Key Highlights
💡 Context-aware predictions using medical conditions
⚙️ End-to-end ML pipeline (preprocessing → prediction)
🌐 Full-stack application (React + Flask)
📈 Realistic dataset including accident and disease cases
📈 Future Improvements
Use real-world large insurance datasets
Improve model accuracy with advanced algorithms
Deploy the application online (AWS / Render / Heroku)
Add user authentication system
Enhance UI/UX design

👩‍💻 Contributors
Qudsia Fatima
Team Members
🎓 Conclusion

This project demonstrates how Machine Learning can be applied to solve real-world problems like insurance fraud detection. It improves efficiency, reduces manual effort, and provides faster and more reliable decision-making.

⭐ If you found this useful, consider giving it a star!