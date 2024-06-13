import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

file_path = "tienxulydulieu_smmh.csv"

smmh = pd.read_csv(file_path)

smmh['Hours Per Day'] = smmh['Hours Per Day'].apply(lambda x: float(x[:-1]))

features_for_clustering = ['Age', 'Hours Per Day', 'ADHD Score', 'Anxiety Score', 
                           'Self Esteem Score', 'Depression Score', 'Total Score']

X = smmh[features_for_clustering]

fig, ax = plt.subplots(2, 2, figsize=(15,10))
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=42)
    km.fit(X)
    silhouette_avg = silhouette_score(X, km.labels_)
    q, mod = divmod(k, 2)
    ax[q-1][mod].plot([silhouette_avg] * len(km.labels_), km.labels_, 'o')
    ax[q-1][mod].set_title(f"K={k}, Silhouette Score={silhouette_avg:.3f}")
    ax[q-1][mod].set_xlabel("Silhouette Score")
    ax[q-1][mod].set_ylabel("Cluster Label")

plt.show()