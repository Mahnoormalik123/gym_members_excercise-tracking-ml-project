import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Title
st.title("Gym Members Exercise Tracking ML Models")

# Load Dataset

st.sidebar.header("User Inputs")
df = pd.read_csv("gym_members_exercise_tracking.csv")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

# Sidebar for user input
st.sidebar.header("User Input Parameters")
age = st.sidebar.slider("Age", 18, 100, 30)
weight = st.sidebar.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
height = st.sidebar.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.75)
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])

# Preprocessing
categorical_columns = ['Gender', 'Workout_Type', 'Experience_Level']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

numerical_features = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 
                    'Resting_BPM', 'Session_Duration (hours)', 'Fat_Percentage', 
                    'Water_Intake (liters)', 'Workout_Frequency (days/week)', 'BMI']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Model Setup
X_regression = df.drop(columns=['Calories_Burned'])
y_regression = df['Calories_Burned']

y_classification = df['Workout_Type_Yoga']  # Example binary target

# Train-Test Split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_regression, y_classification, test_size=0.2, random_state=42)

# Model Selection
model_type = st.sidebar.selectbox("Select Model Type", ["Regression", "Classification"])
model_name = st.sidebar.selectbox("Select Model",["Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN"])

# Regression Models
if model_type == "Regression":
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeRegressor()
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100)
    elif model_name == "SVM":
        model = SVR(kernel='linear')
    elif model_name == "KNN":
        model = KNeighborsRegressor(n_neighbors=5)

    model.fit(X_train_reg, y_train_reg)
    y_pred_reg = model.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    st.write(f"Mean Squared Error (MSE): {mse}")

    # Residuals Plot
    residuals = y_test_reg - y_pred_reg
    sns.histplot(residuals, kde=True, bins=30)
    plt.title("Residuals Distribution")
    st.pyplot(plt)

# Classification Models
elif model_type == "Classification":
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100)
    elif model_name == "SVM":
        model = SVC(kernel='linear')
    elif model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)

    model.fit(X_train_clf, y_train_clf)
    y_pred_clf = model.predict(X_test_clf)
    acc = accuracy_score(y_test_clf, y_pred_clf)
    st.write(f"Accuracy: {acc}")

    # Confusion Matrix
    cm = confusion_matrix(y_test_clf, y_pred_clf)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    st.pyplot(plt)

    # Classification Report
    st.write("Classification Report:")
    st.text(classification_report(y_test_clf, y_pred_clf))
