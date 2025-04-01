import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import networkx as nx
import math

# Config
BLOCK_SIZE = 10000
K = 3
MERGE_THRESHOLD = 0.85

# Step 1: Chunking dataset
def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

# Step 2: Local processing
def local_block_clustering(text_block, model):
    embeddings = model.encode(text_block, convert_to_numpy=True)
    sims = cosine_similarity(embeddings)
    sparse_edges = []
    for i in range(len(text_block)):
        top_k = np.argsort(sims[i])[-K-1:-1][::-1]
        for j in top_k:
            sparse_edges.append((i, j, sims[i][j]))

    G = nx.Graph()
    G.add_nodes_from(range(len(text_block)))
    for i, j, w in sparse_edges:
        G.add_edge(i, j, weight=w)

    clusters = []
    unvisited = set(G.nodes)
    while unvisited:
        current = unvisited.pop()
        neighbors = [n for n in G.neighbors(current) if n in unvisited]
        if not neighbors:
            clusters.append({current})
            continue
        next_node = max(neighbors, key=lambda n: G[current][n]['weight'])
        back_neighbors = [n for n in G.neighbors(next_node) if n in unvisited or n == current]
        back = max(back_neighbors, key=lambda n: G[next_node][n]['weight'], default=None)

        if back == current:
            clusters.append({current, int(next_node)})
            unvisited.discard(next_node)
        else:
            clusters.append({current})
    return clusters, embeddings

# Step 3: Compute centroids of local clusters
def compute_centroids(clusters, embeddings, offset=0):
    centroids = []
    mapping = []
    for cluster in clusters:
        idxs = [i for i in cluster]
        vectors = [embeddings[i] for i in idxs]
        centroid = np.mean(vectors, axis=0)
        centroids.append(centroid)
        mapping.append([i + offset for i in idxs])
    return centroids, mapping

# Step 4: Merge clusters across blocks
def merge_cross_block_clusters(all_centroids, all_mappings, threshold):
    sims = cosine_similarity(all_centroids)
    merged = []
    used = set()
    for i in range(len(sims)):
        if i in used:
            continue
        cluster = set(all_mappings[i])
        for j in range(i+1, len(sims)):
            if j in used:
                continue
            if sims[i][j] >= threshold:
                cluster.update(all_mappings[j])
                used.add(j)
        merged.append(list(cluster))
        used.add(i)
    return merged

# ---- Main Pipeline ----
def teraHAC_clustering(texts):
    model = SentenceTransformer('all-mpnet-base-v2')

    all_centroids = []
    all_mappings = []
    offset = 0

    for block in chunked(texts, BLOCK_SIZE):
        clusters, embeddings = local_block_clustering(block, model)
        centroids, mapping = compute_centroids(clusters, embeddings, offset=offset)
        all_centroids.extend(centroids)
        all_mappings.extend(mapping)
        offset += len(block)

    final_clusters = merge_cross_block_clusters(all_centroids, all_mappings, MERGE_THRESHOLD)
    return final_clusters


from datasets import load_dataset

# Load the English split of the STS Benchmark (Semantic Textual Similarity)
dataset = load_dataset("stsb_multi_mt", name="en", split="train")

# Extract 1000 non-empty sentence1 entries
texts = [ex["sentence1"] for ex in dataset if ex["sentence1"]][:1000]

clusters = teraHAC_clustering(texts)


res = []
for i, cluster in enumerate(clusters):
    #print(f"Final Cluster {i+1}: {[texts[n] for n in cluster]}")
    res.append((len(cluster), [texts[n] for n in cluster]))
    
df = pd.DataFrame(res, columns=["size","sentences"])

df.to_excel("sentences_1000.xlsx", index=False)