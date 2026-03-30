import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # fix for mac ssl issue

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

# load the dataset and train model
# using cache so it doesnt retrain every time the user clicks something
@st.cache_resource
def train_model():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    
    df = pd.read_csv(url, names=col_names)

    # some columns have 0 which doesnt make sense medically so replacing with mean
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df[col] = df[col].replace(0, df[col].mean())

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, scaler, round(acc * 100, 2)


st.title("🩺 Diabetes Risk Predictor")
st.write("Fill in the patient details below and click predict to check diabetes risk.")
st.divider()

with st.spinner("Loading model..."):
    model, scaler, acc = train_model()

st.success(f"Model ready! Accuracy on test data: **{acc}%**")
st.divider()

st.subheader("Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
    bp = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, value=25)

st.divider()

if st.button("Predict", use_container_width=True, type="primary"):
    
    user_input = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    user_input = scaler.transform(user_input)
    
    result = model.predict(user_input)[0]
    proba = model.predict_proba(user_input)[0]
    confidence = round(proba[result] * 100, 1)

    st.subheader("Result")

    if result == 1:
        st.error(f"⚠️ High risk of diabetes detected — {confidence}% confidence")
        st.markdown("""
        **What to do next:**
        - Visit a doctor for proper diagnosis
        - Reduce sugar and processed food intake
        - Try to exercise regularly
        - Monitor blood sugar levels frequently
        """)
    else:
        st.success(f"✅ Low risk of diabetes — {confidence}% confidence")
        st.markdown("""
        **Keep it up:**
        - Maintain a balanced diet
        - Stay active
        - Get yearly health checkups just to be safe
        """)

    # show probability chart
    prob_df = pd.DataFrame({
        "Outcome": ["No Diabetes", "Diabetes"],
        "Probability (%)": [round(proba[0]*100, 1), round(proba[1]*100, 1)]
    })
    st.bar_chart(prob_df.set_index("Outcome"))

    with st.expander("See input summary"):
        st.dataframe(pd.DataFrame({
            "Feature": ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
                        "Insulin", "BMI", "Diabetes Pedigree Function", "Age"],
            "Entered Value": [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
        }), hide_index=True, use_container_width=True)

st.divider()
st.caption("Note: This app is made for a college project and should not be used as actual medical advice.")
