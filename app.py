import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="K-Means Clustering App",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --------------------------
# TITLE
# --------------------------
st.markdown("<h1 style='text-align:center; font-weight:bold; color:#2E86C1;'>K-Means Customer Segmentation</h1>", unsafe_allow_html=True)
st.write("### A simple and interactive clustering app using **Time on Site** and **Pages Visited**.")

# --------------------------
# LOAD MODEL
# --------------------------
try:
    kmeans = joblib.load("kmeans_model.pkl")
    st.success("Model Loaded Successfully ✓")
except:
    st.error("Model not found! Please ensure `kmeans_model.pkl` is in the same folder.")
    st.stop()

# --------------------------
# SIDEBAR USER INPUT
# --------------------------
st.sidebar.markdown("## ⚙️ Input Features")
time_on_site = st.sidebar.slider("Time on Site (minutes)", 0, 60, 10)
pages_visited = st.sidebar.slider("Pages Visited", 1, 30, 5)

user_data = np.array([[time_on_site, pages_visited]])

# --------------------------
# PREDICT CLUSTER
# --------------------------
cluster_label = kmeans.predict(user_data)[0]

st.markdown("## 📊 Prediction Result")
st.info(f"### This user belongs to **Cluster {cluster_label}**")

# --------------------------
# SCATTERPLOT SECTION
# --------------------------
st.markdown("## 🔵 Cluster Visualization")

# Create sample scatter data (optional if you don’t use your CSV)
try:
    df = pd.read_csv("online_shopper_kmeans_dataset.csv")
except:
    st.error("Dataset not found! Please keep the CSV in the same directory.")
    st.stop()

# Predict clusters for entire dataset
df["Cluster"] = kmeans.predict(df[["Time_On_Site_Minutes", "Pages_Visited"]])

# Plot
fig, ax = plt.subplots()
scatter = ax.scatter(
    df["Time_On_Site_Minutes"],
    df["Pages_Visited"],
    c=df["Cluster"]
)

ax.scatter(time_on_site, pages_visited, s=300, edgecolor="black", linewidth=2)
ax.set_xlabel("Time on Site (minutes)")
ax.set_ylabel("Pages Visited")
ax.set_title("K-Means Clustering Visualization")

st.pyplot(fig)

# --------------------------
# FOOTER
# --------------------------
st.write("---")
st.markdown("<p style='text-align:center;'>Developed with ❤️ using Streamlit</p>", unsafe_allow_html=True)
