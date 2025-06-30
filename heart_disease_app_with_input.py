
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

st.set_page_config(page_title="Heart Disease Classifier", layout="wide")

st.title("💓 End-to-End Heart Disease Classification App")

st.write("Upload your dataset (CSV format):")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Target Variable Distribution")
    fig, ax = plt.subplots()
    df['target'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.subheader("Train a Model")
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_choice = st.selectbox("Choose classifier", ["Logistic Regression", "KNN", "Random Forest"])

    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "KNN":
        model = KNeighborsClassifier()
    else:
        model = RandomForestClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.success(f"Model trained! Accuracy: {accuracy:.2f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("📊 Make a Prediction with Custom Input")

    st.write("Enter patient details below to predict the presence of heart disease.")

    input_data = {}
    for col in X.columns:
        if df[col].dtype == 'int64' or df[col].dtype == 'float64':
            val = st.number_input(f"Enter value for {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            input_data[col] = val
        else:
            st.warning(f"Skipping unsupported column: {col}")

    if st.button("Predict Heart Disease"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]

        st.write("### 🩺 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ The model predicts the patient **has heart disease** with probability {prediction_proba:.2f}")
        else:
            st.success(f"✅ The model predicts the patient **does not have heart disease** with probability {1 - prediction_proba:.2f}")
else:
    st.info("Awaiting CSV file upload.")
