
# Customer Segmentation using K-Means Clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. Load data
df = pd.read_csv("Mall_Customers.csv")
print(df.head())
print(df.info())

# 2. Clean data
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
data = df[features].copy()

for col in features:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna().drop_duplicates()

# 3. Scale features
scaler = StandardScaler()
X = scaler.fit_transform(data)

# 4. Find K using Elbow Method and Silhouette Score
inertia = []
silhouette = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(X, labels))

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette, marker="o")
plt.title("Silhouette Score")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.show()

# 5. Final K-Means
K = 3
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
data["Cluster"] = kmeans.fit_predict(X)

# 6. Label clusters using average spending
avg_spending = data.groupby("Cluster")["Spending Score (1-100)"].mean().sort_values()
ordered_clusters = list(avg_spending.index)

labels = ["Low Spenders", "Medium Spenders", "High Spenders"]
label_map = {cluster: labels[i] for i, cluster in enumerate(ordered_clusters)}

data["Segment"] = data["Cluster"].map(label_map)

# 7. Cluster profile
profile = data.groupby(["Cluster", "Segment"])[features].mean().round(2)
print(profile)

# 8. Visualization: Income vs Spending
plt.figure(figsize=(8, 5))
for segment in data["Segment"].unique():
    part = data[data["Segment"] == segment]
    plt.scatter(
        part["Annual Income (k$)"],
        part["Spending Score (1-100)"],
        label=segment,
        alpha=0.75
    )
plt.title("Customer Segmentation: Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

# 9. Visualization: Age vs Spending
plt.figure(figsize=(8, 5))
for segment in data["Segment"].unique():
    part = data[data["Segment"] == segment]
    plt.scatter(
        part["Age"],
        part["Spending Score (1-100)"],
        label=segment,
        alpha=0.75
    )
plt.title("Customer Segmentation: Age vs Spending Score")
plt.xlabel("Age")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

# 10. PCA visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 5))
for segment in data["Segment"].unique():
    mask = data["Segment"] == segment
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=segment, alpha=0.75)

plt.title("PCA Visualization of Customer Segments")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()

# 11. Save result
result = df.loc[data.index].copy()
result["Cluster"] = data["Cluster"].values
result["Segment"] = data["Segment"].values
result.to_csv("customer_segments.csv", index=False)

print("Saved: customer_segments.csv")
