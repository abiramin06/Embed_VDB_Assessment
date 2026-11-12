import os
import time
import itertools
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.cluster import KMeans
from collections import Counter
from google import genai
from google.genai import types

load_dotenv()

sentences = [
    "Python is a popular programming language.",
    "Java is widely used for software development.",
    "Machine learning helps computers learn from data.",
    "Artificial intelligence is transforming technology.",
    "A physician treats patients in a hospital.",
    "A doctor cares for sick people at a clinic.",
    "Pizza is a delicious Italian food.",
    "Pasta is a traditional Italian dish.",
    "Traveling allows people to explore new places.",
    "Tourists enjoy visiting beautiful destinations."
]

topics = [
    "technology", "technology", "technology", "technology",
    "healthcare", "healthcare",
    "food", "food",
    "travel", "travel"
]

print("\n========== LOCAL EMBEDDINGS ==========")

start = time.time()

model = SentenceTransformer("all-MiniLM-L6-v2")
local_embeddings = model.encode(sentences)

local_time = time.time() - start

print("Local embedding shape:", local_embeddings.shape)
print(f"Local embedding time: {local_time:.4f} seconds")

print("\n========== API EMBEDDINGS ==========")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

client = genai.Client(api_key=api_key)

start = time.time()

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=sentences,
    config=types.EmbedContentConfig(
        task_type="SEMANTIC_SIMILARITY"
    )
)

api_embeddings = np.array([
    item.values for item in response.embeddings
])

api_time = time.time() - start

print("API embedding shape:", api_embeddings.shape)
print(f"API embedding time: {api_time:.4f} seconds")

print("\n========== TIME COMPARISON ==========")

print(f"Local model: {local_time:.4f} seconds")
print(f"Gemini API: {api_time:.4f} seconds")

def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


print("\n========== COSINE SIMILARITY TESTS ==========")

tests = [
    ([1, 2, 3], [2, 4, 6]),
    ([1, 0], [0, 1]),
    ([1, 0], [-1, 0])
]

for a, b in tests:
    print(f"{a} vs {b} -> {cosine_similarity(a, b):.1f}")

a = [1, 2, 3]
b = [2, 4, 6]

print(
    "Sklearn result:",
    sklearn_cosine([a], [b])[0][0]
)


print("\n========== SIMILARITY MATRIX ==========")

similarity_matrix = sklearn_cosine(local_embeddings)

print(np.round(similarity_matrix, 2))

print("\n========== TOP 3 SIMILAR PAIRS ==========")

pairs = []

for i, j in itertools.combinations(range(len(sentences)), 2):
    score = similarity_matrix[i][j]
    pairs.append((score, i, j))

pairs.sort(reverse=True)

for score, i, j in pairs[:3]:
    print(f"\nSimilarity: {score:.4f}")
    print("Sentence 1:", sentences[i])
    print("Sentence 2:", sentences[j])

print("\n========== SEMANTIC SEARCH ==========")

queries = [
    "How do computers learn from information?",
    "What food is popular in Italy?",
    "Where can visitors explore new destinations?"
]

for query in queries:

    query_embedding = model.encode([query])

    scores = sklearn_cosine(
        query_embedding,
        local_embeddings
    )[0]

    best_indices = np.argsort(scores)[::-1][:3]

    print(f"\nQuery: {query}")

    for index in best_indices:
        print(
            f"Score: {scores[index]:.4f} | "
            f"{sentences[index]}"
        )
print("\n========== DUPLICATE DETECTION ==========")

threshold = 0.70

for score, i, j in pairs:

    if score >= threshold:
        print(
            f"Possible duplicate ({score:.4f}): "
            f"\n{sentences[i]}"
            f"\n{sentences[j]}\n"
        )

print("\n========== K-MEANS CLUSTERING ==========")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_ids = kmeans.fit_predict(local_embeddings)

for cluster_id in sorted(set(cluster_ids)):

    indices = np.where(cluster_ids == cluster_id)[0]

    cluster_topics = [
        topics[index] for index in indices
    ]

    cluster_name = Counter(cluster_topics).most_common(1)[0][0]

    print(f"\nCluster {cluster_id}: {cluster_name}")

    for index in indices:
        print("-", sentences[index])

print("\n========== SEMANTIC SIMILARITY EXAMPLE ==========")

i, j = 4, 5

print("Sentence 1:", sentences[i])
print("Sentence 2:", sentences[j])

print(
    "Similarity score:",
    round(float(similarity_matrix[i][j]), 4)
)

print("\n========== LOCAL VS API ==========")

print("1. Local models offer privacy and offline access.")
print("2. API models avoid local hardware requirements.")
print("3. Local models have no per-request API cost.")
print("4. API models may offer higher quality and easy scaling.")

print("\nTASK 1 EXECUTION COMPLETED")