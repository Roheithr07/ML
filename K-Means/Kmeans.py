# K-Means Clustering using Mall Customers Dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Display first 5 rows
print("Dataset:")
print(data.head())

# Select features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()

# Apply K-Means with K = 5
kmeans = KMeans(n_clusters=5, init='k-means++',
                random_state=42, n_init=10)

# Predict clusters
data['Cluster'] = kmeans.fit_predict(X)

# Display clustered data
print("\nClustered Dataset:")
print(data.head(10))

# Plot clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    X.iloc[data['Cluster'] == 0, 0],
    X.iloc[data['Cluster'] == 0, 1],
    label='Cluster 1'
)

plt.scatter(
    X.iloc[data['Cluster'] == 1, 0],
    X.iloc[data['Cluster'] == 1, 1],
    label='Cluster 2'
)

plt.scatter(
    X.iloc[data['Cluster'] == 2, 0],
    X.iloc[data['Cluster'] == 2, 1],
    label='Cluster 3'
)

plt.scatter(
    X.iloc[data['Cluster'] == 3, 0],
    X.iloc[data['Cluster'] == 3, 1],
    label='Cluster 4'
)

plt.scatter(
    X.iloc[data['Cluster'] == 4, 0],
    X.iloc[data['Cluster'] == 4, 1],
    label='Cluster 5'
)

# Plot centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker='X',
    label='Centroids'
)

plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()