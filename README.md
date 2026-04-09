# ✈️ Flight Delay Prediction System

## 📌 Overview
Flight delays are a major issue in the aviation industry, affecting passengers and airline operations.  
This project uses **machine learning in R** to predict whether a flight will be **Delayed** or **On-Time** using historical flight data.

We implemented:
- Logistic Regression
- Random Forest
- XGBoost

An interactive **Shiny web application** is also developed for real-time predictions.

---

## 🚀 Features
- Data preprocessing and feature engineering  
- Binary classification (Delayed / On-Time)  
- Multiple ML models for comparison  
- Performance evaluation (Accuracy, Precision, Recall, F1 Score, AUC)  
- Interactive **Shiny GUI**

---

## 📊 Correlation Heatmap
The heatmap shows relationships between features.  
**Departure delay (DepDelay)** has the strongest influence on arrival delay.

<img width="413" height="412" alt="Screenshot 2026-04-09 at 11 20 30 AM" src="https://github.com/user-attachments/assets/eb721021-cc00-4663-a036-a83c22635cdc" />



---

## 📈 Model Performance Comparison
This chart compares all models across evaluation metrics.

- Random Forest → Best overall performance  
- XGBoost → Best AUC & Precision  
- Logistic Regression → Strong baseline  

<img width="410" height="483" alt="Screenshot 2026-04-09 at 11 20 50 AM" src="https://github.com/user-attachments/assets/d27e597a-bc86-4f48-889b-c0708c254c0e" />


---

## 🖥️ Shiny GUI (Application)

### 🔴 Delayed Flight Prediction
Example where the model predicts a delay with high probability:

<img width="416" height="257" alt="Screenshot 2026-04-09 at 11 21 29 AM" src="https://github.com/user-attachments/assets/ce5c033f-2d22-4beb-b36f-ecbaee720e82" />


---

### 🟢 On-Time Flight Prediction
Example where the model predicts no delay:

<img width="412" height="331" alt="Screenshot 2026-04-09 at 11 21 47 AM" src="https://github.com/user-attachments/assets/5b005e50-22da-478c-9688-26a480553ed7" />


---

## 🧠 Key Insights
- Departure delay is the strongest predictor  
- Weather and late aircraft delays significantly impact outcomes  
- Distance has minimal effect  
- Feature engineering improves model performance  

---
