# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generating sample data
# Make 300 samples with 2 features (X) and 4 centers (clusters)
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Visualizing the generated data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=50, cmap='viridis')
plt.title('Generated Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Applying KMeans algorithm
kmeans = KMeans(n_clusters=4, random_state=0)  # Creating a KMeans instance with 4 clusters
kmeans.fit(X)  # Fitting the model to the data

# Getting cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Visualizing the clustered data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', alpha=0.5, label='Data points')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, label='Cluster centers')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()