# 🩺 Diabetes Risk Predictor

A machine learning web app that predicts whether a person is at risk of diabetes based on basic health measurements. Built with Python, scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red) ![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 What It Does

You enter 8 basic health values (like glucose level, BMI, and age) and the app tells you whether you have a **high or low risk of diabetes**, along with:

- A **confidence percentage** for the prediction
- A **probability chart** showing the likelihood of each outcome
- **Health recommendations** based on the result

The model trains automatically when you first open the app — no manual setup or separate training script needed.

---

## 🖥️ Demo

| Low Risk Result | High Risk Result |
|---|---|
| ✅ Green banner with healthy tips | ⚠️ Red banner with next steps |

---

## 🤖 How It Works

1. The app downloads the **Pima Indians Diabetes Dataset** (768 patient records) on first run
2. Missing values (stored as 0s in the dataset) are replaced with column means
3. Features are scaled using **StandardScaler**
4. A **Random Forest Classifier** (100 trees) is trained on 80% of the data
5. The remaining 20% is used to evaluate accuracy (~77–80%)
6. Your inputs are passed through the same scaler and into the trained model
7. The result is displayed along with prediction probabilities

---

## 📦 Requirements

- Python 3.8 or higher
- pip or pip3

All dependencies are listed in `requirements.txt`:

```
streamlit
pandas
numpy
scikit-learn
```

---

## 🚀 Getting Started

### Step 1 — Clone the repository

```bash
git clone https://github.com/HyperSpandan/diabetes-predictor.git
cd diabetes-predictor
```

### Step 2 — Install dependencies

```bash
pip3 install -r requirements.txt
```

> On Windows, you may use `pip` instead of `pip3`

### Step 3 — Run the app

```bash
python3 -m streamlit run app.py
```

> On Windows: `python -m streamlit run app.py`

Your browser will automatically open at **http://localhost:8501**

---

## 🗂️ Project Structure

```
diabetes-predictor/
│
├── app.py              # Main Streamlit web application
├── requirements.txt    # Python dependencies
└── README.md           # You are here
```

---

## 📊 Dataset

**Pima Indians Diabetes Dataset**

- **Source:** UCI Machine Learning Repository / Kaggle
- **Samples:** 768 patient records
- **Features:** 8 input features + 1 target variable (diabetic or not)
- **Origin:** National Institute of Diabetes and Digestive and Kidney Diseases

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (mg/dL) |
| Blood Pressure | Diastolic blood pressure (mm Hg) |
| Skin Thickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (μU/mL) |
| BMI | Body mass index (kg/m²) |
| Diabetes Pedigree Function | Genetic risk score based on family history |
| Age | Age in years |

---

## ⚙️ Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Number of Trees | 100 |
| Train / Test Split | 80% / 20% |
| Test Accuracy | ~77–80% |
| Preprocessing | Mean imputation + StandardScaler |

---

## 🐛 Known Issues & Fixes

### SSL Error on macOS (Python 3.13)

If you see an `SSL: CERTIFICATE_VERIFY_FAILED` error, it means your Mac's Python installation is missing SSL certificates. This is already handled in the code with:

```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

No action needed — the fix is already in `app.py`.

### "command not found: streamlit" or "command not found: pip"

Use the full Python module path instead:

```bash
python3 -m streamlit run app.py   # instead of: streamlit run app.py
pip3 install -r requirements.txt  # instead of: pip install ...
```

---

## 🔮 Possible Future Improvements

- Compare multiple models (Logistic Regression, SVM, KNN) side by side
- Add a feature importance chart
- Support CSV upload for batch predictions
- Deploy to Streamlit Cloud for public access
- Handle class imbalance using SMOTE

---

## ⚠️ Disclaimer

This app is built for a **college AI/ML project** and is intended for educational purposes only. It is not a substitute for professional medical advice. Always consult a qualified doctor for any health concerns.

---

## 👤 Author

**Spandan Dhage**

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
