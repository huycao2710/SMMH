import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data
filepath = "tienxulydulieu_smmh.csv"
smmh = pd.read_csv(filepath)

# Preprocess data
smmh['Hours Per Day'] = smmh['Hours Per Day'].apply(lambda x: float(x[:-1]))

# Select features for clustering
features_for_clustering = ['Age', 'Hours Per Day', 'ADHD Score', 'Anxiety Score', 
                           'Self Esteem Score', 'Depression Score', 'Total Score']

X = smmh[features_for_clustering]

# Function to analyze clusters
def analyze_clusters(X, labels, k):
    for i in range(k):
        cluster_data = X[labels == i]
        print(f"Cluster {i+1}:")
        print(f"  Age: {cluster_data['Age'].mean():.2f} ± {cluster_data['Age'].std():.2f}")
        print(f"  Hours Per Day: {cluster_data['Hours Per Day'].mean():.2f} ± {cluster_data['Hours Per Day'].std():.2f}")
        print(f"  ADHD Score: {cluster_data['ADHD Score'].mean():.2f} ± {cluster_data['ADHD Score'].std():.2f}")
        print(f"  Anxiety Score: {cluster_data['Anxiety Score'].mean():.2f} ± {cluster_data['Anxiety Score'].std():.2f}")
        print(f"  Self Esteem Score: {cluster_data['Self Esteem Score'].mean():.2f} ± {cluster_data['Self Esteem Score'].std():.2f}")
        print(f"  Depression Score: {cluster_data['Depression Score'].mean():.2f} ± {cluster_data['Depression Score'].std():.2f}")
        print(f"  Total Score: {cluster_data['Total Score'].mean():.2f} ± {cluster_data['Total Score'].std():.2f}")
        print()

# Function to visualize clusters
def visualize_clusters(X, labels, k, centroids):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X['Age'], y=X['Hours Per Day'], hue=labels, palette="viridis")
    plt.title(f"K-Means Clustering with k={k}")
    plt.xlabel("Age")
    plt.ylabel("Hours Per Day")
    
    # Add cluster names and centroids
    for i, centroid in enumerate(centroids):
        plt.scatter(centroid[0], centroid[1], marker="X", s=200, linewidth=2, color="black")
        plt.annotate(f"Cluster {i+1}", (centroid[0], centroid[1]), textcoords="offset points", xytext=(0, 10), ha='center')
    
    plt.show()


# Run K-Means clustering with k=2
k = 5
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

visualize_clusters(X, labels, k, centroids)
analyze_clusters(X, labels, k)