import json
import time
import uuid
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


DATA_FILE = Path("data/T2/metadata_chunks.jsonl")
QDRANT_DIR = Path("data/T3/qdrant_db")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384
COLLECTION_NAME = "policy_documents"

print("\n========== ANN AND HNSW ==========")

print("1. ANN means Approximate Nearest Neighbour search.")
print("2. ANN finds similar vectors without checking every vector.")
print("3. HNSW means Hierarchical Navigable Small World.")
print("4. HNSW builds graph connections between nearby vectors.")
print("5. It provides fast search with a small possible recall loss.")

print("\n========== FAISS EXPERIMENT ==========")

np.random.seed(42)

NUM_VECTORS = 50000
DIMENSIONS = 384
NUM_QUERIES = 100
TOP_K = 10

# Generate random vectors
database = np.random.random(
    (NUM_VECTORS, DIMENSIONS)
).astype("float32")

queries = np.random.random(
    (NUM_QUERIES, DIMENSIONS)
).astype("float32")

# Normalize vectors for cosine similarity
faiss.normalize_L2(database)
faiss.normalize_L2(queries)

# Exact search
exact_index = faiss.IndexFlatIP(DIMENSIONS)
exact_index.add(database)

start = time.perf_counter()

exact_scores, exact_results = exact_index.search(
    queries, TOP_K
)

exact_time = time.perf_counter() - start

# HNSW approximate search
hnsw_index = faiss.IndexHNSWFlat(
    DIMENSIONS,
    32,
    faiss.METRIC_INNER_PRODUCT
)

hnsw_index.hnsw.efConstruction = 100
hnsw_index.hnsw.efSearch = 64

hnsw_index.add(database)

start = time.perf_counter()

hnsw_scores, hnsw_results = hnsw_index.search(
    queries, TOP_K
)

hnsw_time = time.perf_counter() - start

# Calculate recall@10
recall_values = []

for exact, approximate in zip(exact_results, hnsw_results):

    recall = len(
        set(exact) & set(approximate)
    ) / TOP_K

    recall_values.append(recall)

recall_at_10 = np.mean(recall_values)

print(f"Vectors: {NUM_VECTORS}")
print(f"Dimensions: {DIMENSIONS}")
print(f"Exact search time: {exact_time:.4f} seconds")
print(f"HNSW search time: {hnsw_time:.4f} seconds")
print(f"Recall@10: {recall_at_10:.4f}")


print("\n========== LOADING DOCUMENT CHUNKS ==========")

chunks = []

with open(DATA_FILE, "r", encoding="utf-8") as file:

    for line in file:
        chunks.append(json.loads(line))

print("Total document chunks:", len(chunks))


print("\n========== LOADING EMBEDDING MODEL ==========")

model = SentenceTransformer(MODEL_NAME)

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)

print("Embedding shape:", embeddings.shape)
print("Vector dimensions:", embeddings.shape[1])


print("\n========== QDRANT COLLECTION ==========")

QDRANT_DIR.mkdir(parents=True, exist_ok=True)

client = QdrantClient(
    path=str(QDRANT_DIR)
)

# Recreate collection for a clean assessment run
if client.collection_exists(COLLECTION_NAME):

    client.delete_collection(COLLECTION_NAME)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=VECTOR_SIZE,
        distance=models.Distance.COSINE
    ),
    hnsw_config=models.HnswConfigDiff(
        m=16,
        ef_construct=100
    )
)

print("Qdrant collection created successfully.")
print("Distance: Cosine")
print("Vector size: 384")
print("HNSW M: 16")
print("HNSW ef_construct: 100")


client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="doc_type",
    field_schema=models.PayloadSchemaType.KEYWORD
)

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="page",
    field_schema=models.PayloadSchemaType.INTEGER
)

print("Payload indexes created.")

print("\n========== QDRANT UPSERT ==========")

points = []

for chunk, embedding in zip(chunks, embeddings):

    metadata = chunk["metadata"]

    # Convert SHA-256 ID to deterministic UUID
    point_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            chunk["id"]
        )
    )

    payload = {
        "text": chunk["text"],
        "source": metadata["source"],
        "page": int(metadata["page"]),
        "section_heading": metadata["section_heading"],
        "created_date": metadata["created_date"],
        "chunk_index": metadata["chunk_index"],
        "doc_type": metadata["doc_type"]
    }

    points.append(
        models.PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=payload
        )
    )

# First upsert
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
    wait=True
)

count1 = client.count(
    collection_name=COLLECTION_NAME,
    exact=True
).count

print("Run 1:", count1, "points")

# Second upsert
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
    wait=True
)

count2 = client.count(
    collection_name=COLLECTION_NAME,
    exact=True
).count

print("Run 2:", count2, "points")
print("Count unchanged:", count1 == count2)

print("\n========== FILTERED SEARCH ==========")

query = "Government policy and annual report"

query_vector = model.encode(
    query,
    normalize_embeddings=True
).tolist()

search_filter = models.Filter(
    must=[
        models.FieldCondition(
            key="doc_type",
            match=models.MatchValue(value="pdf")
        ),
        models.FieldCondition(
            key="page",
            range=models.Range(gte=3)
        )
    ]
)

search_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    query_filter=search_filter,
    limit=5,
    with_payload=True
).points

print("Filter: doc_type=pdf, page >= 3")
print("Results found:", len(search_results))

for result in search_results:

    print("\nSource:", result.payload["source"])
    print("Page:", result.payload["page"])
    print("Score:", round(result.score, 4))
    print("Text:", result.payload["text"][:150])


print("\n========== DELETE BY SOURCE ==========")

source_to_delete = chunks[0]["metadata"]["source"]

client.delete(
    collection_name=COLLECTION_NAME,
    points_selector=models.FilterSelector(
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(
                        value=source_to_delete
                    )
                )
            ]
        )
    ),
    wait=True
)

remaining = client.count(
    collection_name=COLLECTION_NAME,
    exact=True
).count

print("Deleted source:", source_to_delete)
print("Remaining points:", remaining)

print("\n========== PGVECTOR VS QDRANT ==========")

print(
    "\nTechnology | Setup | Filtering | Best use"
    "\npgvector   | PostgreSQL extension | SQL filters | Existing SQL systems"
    "\nQdrant     | Dedicated vector DB   | Payload filters | Semantic search applications"
)

print("\nQDRANT COLLECTION INFORMATION")

info = client.get_collection(COLLECTION_NAME)

print("Points:", remaining)
print("HNSW configuration:", info.config.hnsw_config)

print("\nTASK 3 VECTOR STORAGE AND ANN COMPLETED")

client.close()
del client
