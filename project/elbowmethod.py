import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

file_path = "tienxulydulieu_smmh.csv"

smmh = pd.read_csv(file_path)

smmh['Hours Per Day'] = smmh['Hours Per Day'].apply(lambda x: float(x[:-1]))

features_for_clustering = ['Age', 'Hours Per Day', 'ADHD Score', 'Anxiety Score', 
                           'Self Esteem Score', 'Depression Score', 'Total Score']

X = smmh[features_for_clustering]

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#k=3 / k=4