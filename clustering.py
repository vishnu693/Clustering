import os
import numpy as np
import networkx as nx
import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from datasets import load_dataset
import requests
import zipfile

# --- Config ---
BLOCK_SIZE = 10000
K = 3
MERGE_THRESHOLD = 0.85
GLOVE_DIM = 300
GLOVE_ZIP_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILENAME = "glove.6B.300d.txt"
GLOVE_DIR = "glove_data"
GLOVE_PATH = os.path.join(GLOVE_DIR, GLOVE_FILENAME)

# --- Download GloVe if not present ---
def ensure_glove():
    if not os.path.exists(GLOVE_PATH):
        os.makedirs(GLOVE_DIR, exist_ok=True)
        zip_path = os.path.join(GLOVE_DIR, "glove.6B.zip")
        print("Downloading GloVe embeddings...")
        r = requests.get(GLOVE_ZIP_URL, stream=True)
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Extracting GloVe...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(GLOVE_FILENAME, GLOVE_DIR)
        os.remove(zip_path)

# --- Load GloVe Vectors ---
def load_glove(path):
    word_vecs = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=float)
            word_vecs[word] = vec
    return word_vecs

ensure_glove()
glove_vectors = load_glove(GLOVE_PATH)

# --- TF-IDF Weighted Sentence Embeddings ---
def compute_weighted_embeddings(texts, word_vecs, dim=300):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)
    vocab = vectorizer.vocabulary_
    idf = vectorizer.idf_
    idf_dict = {word: idf[i] for word, i in vocab.items()}

    embeddings = []
    for text in texts:
        words = text.lower().split()
        vecs = []
        weights = []
        for word in words:
            if word in word_vecs and word in idf_dict:
                vecs.append(word_vecs[word])
                weights.append(idf_dict[word])
        if vecs:
            weighted = np.average(vecs, axis=0, weights=weights)
        else:
            weighted = np.zeros(dim)
        embeddings.append(weighted)
    return np.array(embeddings)

# --- PCA Dimensionality Reduction ---
def reduce_embeddings(embeddings, out_dim=100):
    pca = PCA(n_components=out_dim)
    return pca.fit_transform(embeddings)

# --- Chunking ---
def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

# --- Local Clustering ---
def local_block_clustering(text_block, embeddings):
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
    return clusters

# --- Compute Centroids ---
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

# --- Merge Clusters Across Blocks ---
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

# --- Main TeraHAC Pipeline ---
def teraHAC_clustering(texts):
    embeddings = compute_weighted_embeddings(texts, glove_vectors)
    reduced_embeddings = reduce_embeddings(embeddings, out_dim=100)

    all_centroids = []
    all_mappings = []
    offset = 0

    for block in chunked(list(zip(texts, reduced_embeddings)), BLOCK_SIZE):
        block_texts = [b[0] for b in block]
        block_embeds = np.array([b[1] for b in block])
        clusters = local_block_clustering(block_texts, block_embeds)
        centroids, mapping = compute_centroids(clusters, block_embeds, offset=offset)
        all_centroids.extend(centroids)
        all_mappings.extend(mapping)
        offset += len(block)

    final_clusters = merge_cross_block_clusters(all_centroids, all_mappings, MERGE_THRESHOLD)
    return final_clusters