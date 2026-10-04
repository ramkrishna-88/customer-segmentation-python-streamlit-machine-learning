
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Customer Segmentation using Clustering")
st.caption("K-Means • Mall Customers • Business Analytics")

st.sidebar.header("Project Controls")
uploaded = st.sidebar.file_uploader(
    "Upload Mall Customers CSV",
    type=["csv"],
    help="Use the Kaggle Mall Customers Dataset or the included Mall_Customers.csv."
)

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("Mall_Customers.csv")

# Normalize common Kaggle column names.
rename_map = {}
for c in df.columns:
    clean = c.strip()
    if clean.lower() == "customerid":
        rename_map[c] = "CustomerID"
    elif clean.lower() == "gender":
        rename_map[c] = "Gender"
    elif clean.lower() == "age":
        rename_map[c] = "Age"
    elif "annual income" in clean.lower():
        rename_map[c] = "Annual Income (k$)"
    elif "spending score" in clean.lower():
        rename_map[c] = "Spending Score (1-100)"

df = df.rename(columns=rename_map)

required = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
missing = [c for c in required if c not in df.columns]

if missing:
    st.error(
        "Missing required columns: " + ", ".join(missing) +
        ". Expected Age, Annual Income (k$), and Spending Score (1-100)."
    )
    st.stop()

# Data cleaning
work = df.copy()
for c in required:
    work[c] = pd.to_numeric(work[c], errors="coerce")

before = len(work)
work = work.dropna(subset=required).drop_duplicates()
removed = before - len(work)

st.sidebar.write(f"Rows after cleaning: {len(work)}")
if removed:
    st.sidebar.info(f"Removed {removed} invalid/duplicate row(s).")

st.subheader("1. Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", len(work))
c2.metric("Avg Age", f"{work['Age'].mean():.1f}")
c3.metric("Avg Income", f"{work['Annual Income (k$)'].mean():.1f} k$")
c4.metric("Avg Spending", f"{work['Spending Score (1-100)'].mean():.1f}")

with st.expander("View cleaned data"):
    st.dataframe(work, use_container_width=True)

st.subheader("2. Exploratory Data Analysis")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    ax.hist(work["Age"], bins=12)
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Customers")
    st.pyplot(fig, clear_figure=True)

with col2:
    fig, ax = plt.subplots()
    ax.scatter(
        work["Annual Income (k$)"],
        work["Spending Score (1-100)"],
        alpha=0.75
    )
    ax.set_title("Income vs Spending Score")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score")
    st.pyplot(fig, clear_figure=True)

st.subheader("3. Choose Number of Clusters")

max_k = min(10, len(work) - 1)
default_k = min(3, max_k)
k = st.sidebar.slider("Number of clusters (K)", 2, max_k, default_k)

features = work[required].copy()

# Scaling is important because Age, Income and Spending Score have different ranges.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Elbow curve
inertias = []
silhouettes = []
ks = range(2, max_k + 1)

for kk in ks:
    model = KMeans(n_clusters=kk, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

e1, e2 = st.columns(2)
with e1:
    fig, ax = plt.subplots()
    ax.plot(list(ks), inertias, marker="o")
    ax.set_title("Elbow Method")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    st.pyplot(fig, clear_figure=True)

with e2:
    fig, ax = plt.subplots()
    ax.plot(list(ks), silhouettes, marker="o")
    ax.set_title("Silhouette Score")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette Score")
    st.pyplot(fig, clear_figure=True)

st.caption(
    "The assignment workflow uses K-Means. The elbow and silhouette plots are "
    "included to help inspect a reasonable K before final segmentation."
)

# Final K-Means model
model = KMeans(n_clusters=k, random_state=42, n_init=10)
work["Cluster"] = model.fit_predict(X_scaled)

# Label clusters according to their average spending score.
cluster_spending = work.groupby("Cluster")["Spending Score (1-100)"].mean().sort_values()
tier_names = ["Low Spenders", "Medium Spenders", "High Spenders"]

if k == 3:
    tier_map = {cluster: tier_names[i] for i, cluster in enumerate(cluster_spending.index)}
else:
    # For K other than 3, retain descriptive spending-order labels.
    ordered = list(cluster_spending.index)
    tier_map = {}
    for i, cluster in enumerate(ordered):
        if i == 0:
            tier_map[cluster] = "Low Spenders"
        elif i == len(ordered) - 1:
            tier_map[cluster] = "High Spenders"
        else:
            tier_map[cluster] = "Medium Spenders"

work["Segment"] = work["Cluster"].map(tier_map)

st.subheader("4. Customer Segments")

segments = work["Segment"].value_counts()
m1, m2, m3 = st.columns(3)
m1.metric("Low Spenders", int(segments.get("Low Spenders", 0)))
m2.metric("Medium Spenders", int(segments.get("Medium Spenders", 0)))
m3.metric("High Spenders", int(segments.get("High Spenders", 0)))

# 2D visualization: income vs spending
fig, ax = plt.subplots(figsize=(8, 5))
for segment in sorted(work["Segment"].unique()):
    part = work[work["Segment"] == segment]
    ax.scatter(
        part["Annual Income (k$)"],
        part["Spending Score (1-100)"],
        label=segment,
        alpha=0.75
    )
ax.set_title("Customer Groups: Income vs Spending Score")
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.legend()
st.pyplot(fig, clear_figure=True)

# Age vs spending
fig, ax = plt.subplots(figsize=(8, 5))
for segment in sorted(work["Segment"].unique()):
    part = work[work["Segment"] == segment]
    ax.scatter(
        part["Age"],
        part["Spending Score (1-100)"],
        label=segment,
        alpha=0.75
    )
ax.set_title("Customer Groups: Age vs Spending Score")
ax.set_xlabel("Age")
ax.set_ylabel("Spending Score (1-100)")
ax.legend()
st.pyplot(fig, clear_figure=True)

# 3D plot
try:
    import plotly.express as px
    fig3d = px.scatter_3d(
        work,
        x="Age",
        y="Annual Income (k$)",
        z="Spending Score (1-100)",
        color="Segment",
        hover_data=["CustomerID"] if "CustomerID" in work.columns else None,
        title="3D Customer Segmentation"
    )
    st.plotly_chart(fig3d, use_container_width=True)
except Exception:
    st.info("Install plotly to display the interactive 3D visualization.")

st.subheader("5. Cluster Profiles")
profile = (
    work.groupby(["Cluster", "Segment"])[required]
    .mean()
    .round(2)
    .sort_values("Spending Score (1-100)")
)
st.dataframe(profile, use_container_width=True)

st.subheader("6. PCA Visualization")
pca = PCA(n_components=2, random_state=42)
pca_data = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame({
    "PC1": pca_data[:, 0],
    "PC2": pca_data[:, 1],
    "Segment": work["Segment"].values
})

fig, ax = plt.subplots(figsize=(8, 5))
for segment in sorted(pca_df["Segment"].unique()):
    part = pca_df[pca_df["Segment"] == segment]
    ax.scatter(part["PC1"], part["PC2"], label=segment, alpha=0.75)
ax.set_title("PCA View of Customer Segments")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend()
st.pyplot(fig, clear_figure=True)

st.subheader("7. Business Interpretation")
st.markdown("""
- **Low Spenders:** customers with relatively low average spending scores. They can be
  studied for retention offers, introductory promotions, or product discovery campaigns.
- **Medium Spenders:** customers with moderate spending behavior. Personalized bundles
  and cross-selling can be explored.
- **High Spenders:** customers with high spending scores. Loyalty programs and
  personalized premium offers can be considered.

These labels are descriptive cluster names based on average spending score; they are
not predictions of individual future spending.
""")

csv_bytes = work.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Segmented Customer Data",
    data=csv_bytes,
    file_name="customer_segments.csv",
    mime="text/csv"
)

st.success("Customer segmentation completed successfully.")
