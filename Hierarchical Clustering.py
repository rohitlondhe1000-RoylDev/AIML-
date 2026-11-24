import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y_true = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

linked = linkage(X_scaled, method="ward")

hc = AgglomerativeClustering(n_clusters=3, linkage="ward")
hc_labels = hc.fit_predict(X_scaled)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
km_labels = kmeans.fit_predict(X_scaled)

hc_sil = silhouette_score(X_scaled, hc_labels)
km_sil = silhouette_score(X_scaled, km_labels)

hc_ari = adjusted_rand_score(y_true, hc_labels)
km_ari = adjusted_rand_score(y_true, km_labels)

print("--- Clustering Performance Comparison ---")
print(f"Hierarchical Clustering -> Silhouette Score: {hc_sil:.4f} | Adjusted Rand Index: {hc_ari:.4f}")
print(f"K-Means Clustering      -> Silhouette Score: {km_sil:.4f} | Adjusted Rand Index: {km_ari:.4f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
dendrogram(linked, truncate_mode="lastp", p=30, leaf_rotation=90.0, leaf_font_size=8.0)
plt.axhline(y=10, color="red", linestyle="--", label="Cut Threshold (3 Clusters)")
plt.title("Hierarchical Clustering Dendrogram (Ward)")
plt.xlabel("Sample Index / Cluster Size")
plt.ylabel("Euclidean Distance")
plt.legend()

plt.subplot(1, 3, 2)
sns.scatterplot(x=X_scaled[:, 2], y=X_scaled[:, 3], hue=hc_labels, palette="viridis", s=70)
plt.title("Hierarchical Clusters (Petal Length vs Width)")
plt.xlabel("Standardized Petal Length")
plt.ylabel("Standardized Petal Width")

plt.subplot(1, 3, 3)
sns.scatterplot(x=X_scaled[:, 2], y=X_scaled[:, 3], hue=km_labels, palette="plasma", s=70)
plt.title("K-Means Clusters (Petal Length vs Width)")
plt.xlabel("Standardized Petal Length")
plt.ylabel("Standardized Petal Width")

plt.tight_layout()
plt.show()




#--- Clustering Performance Comparison ---
#Hierarchical Clustering -> Silhouette Score: 0.4467 | Adjusted Rand Index: 0.6153
#K-Means Clustering      -> Silhouette Score: 0.4599 | Adjusted Rand Index: 0.6201