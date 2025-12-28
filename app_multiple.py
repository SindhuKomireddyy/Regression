import streamlit as st
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")   # REQUIRED for Streamlit Cloud
import matplotlib.pyplot as plt
import seaborn as sns

import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Multiple Linear Regression",
    layout="centered"
)

# ---------------- SAFE CSS LOADER ----------------
def load_css():
    css_path = os.path.join(os.getcwd(), "style.css")
    if os.path.isfile(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()   # ✅ SAFE CALL (NO ARGUMENTS)


# ---------------- TITLE ----------------
st.markdown("""
<div class="card">
    <h1>Multiple Linear Regression</h1>
    <p>Predict <b>Tip Amount</b> using multiple features</p>
</div>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()


# ---------------- DATASET PREVIEW ----------------
st.subheader("Dataset Preview")
st.dataframe(df.head())


# ---------------- DATA PREPARATION ----------------
X = df[["total_bill", "size"]]
y = df["tip"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---------------- MODEL TRAINING ----------------
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# ---------------- METRICS ----------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X.shape[1] - 1)


# ---------------- VISUALIZATION ----------------
st.subheader("Total Bill vs Tip Amount")

fig, ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip"], alpha=0.6)
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip Amount")
st.pyplot(fig)


# ---------------- PERFORMANCE ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R² Score", f"{r2:.3f}")
c4.metric("Adjusted R²", f"{adj_r2:.3f}")

st.markdown("</div>", unsafe_allow_html=True)


# ---------------- COEFFICIENTS ----------------
st.markdown(f"""
<div class="card">
    <h2>Model Coefficients</h2>
    <p>
        <b>Total Bill Coefficient:</b> {model.coef_[0]:.3f}<br>
        <b>Table Size Coefficient:</b> {model.coef_[1]:.3f}<br>
        <b>Intercept:</b> {model.intercept_:.3f}
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- PREDICTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Amount")

bill = st.slider(
    "Total Bill",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)

size = st.slider(
    "Table Size",
    int(df.size.min()),
    int(df.size.max()),
    2
)

input_data = scaler.transform([[bill, size]])
predicted_tip = model.predict(input_data)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Tip: ${predicted_tip:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)
