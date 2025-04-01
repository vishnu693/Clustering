from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

subset = texts[:1000]
X = compute_weighted_embeddings(subset, glove_vectors)
X_reduced = reduce_embeddings(X, out_dim=100)

hac = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - MERGE_THRESHOLD, affinity='cosine', linkage='average')
hac_labels = hac.fit_predict(X_reduced)

tera_clusters = teraHAC_clustering(subset)
tera_labels = np.zeros(len(subset))
for i, cluster in enumerate(tera_clusters):
    for idx in cluster:
        tera_labels[idx] = i

# Compare clusters
from sklearn.metrics import adjusted_rand_score
score = adjusted_rand_score(hac_labels, tera_labels)
print(\"Adjusted Rand Index:\", score)
