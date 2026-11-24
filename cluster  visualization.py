import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
columns = [
    "Class", "Alcohol", "Malic_Acid", "Ash", "Alcalinity_of_Ash",
    "Magnesium", "Total_Phenols", "Flavanoids", "Nonflavanoid_Phenols",
    "Proanthocyanins", "Color_Intensity", "Hue", "OD280_OD315", "Proline"
]

df = pd.read_csv(url, names=columns)
X = df.drop(columns=["Class"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
silhouette_scores = []
k_range = range(2, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

optimal_k = 3
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = final_kmeans.fit_predict(X_scaled)
final_silhouette = silhouette_score(X_scaled, clusters)

print(f"Dataset Shape (Features) : {X.shape}")
print(f"Optimal Clusters (K)     : {optimal_k}")
print(f"Silhouette Score (K=3)   : {final_silhouette:.4f}")
print(f"Within-Cluster Inertia   : {final_kmeans.inertia_:.2f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(final_kmeans.cluster_centers_)

plt.figure(figsize=(15, 4.5))

plt.subplot(1, 3, 1)
plt.plot(k_range, inertias, marker="o", color="steelblue")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method for Optimal K")
plt.grid(True, linestyle="--", alpha=0.5)

plt.subplot(1, 3, 2)
plt.plot(k_range, silhouette_scores, marker="s", color="darkorange")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs. K")
plt.grid(True, linestyle="--", alpha=0.5)

plt.subplot(1, 3, 3)
sns.scatterplot(
    x=X_pca[:, 0], y=X_pca[:, 1],
    hue=clusters, palette="viridis", s=70, alpha=0.85
)
plt.scatter(
    centroids_pca[:, 0], centroids_pca[:, 1],
    c="red", s=200, marker="X", label="Centroids"
)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
plt.title("K-Means Clusters Projected in 2D (PCA)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()




#Dataset Shape (Features) : (178, 13)
#Optimal Clusters (K)     : 3
#Silhouette Score (K=3)   : 0.2849
#Within-Cluster Inertia   : 1277.93
#PS C:\Users\91976\Desktop\AIML LAB> 