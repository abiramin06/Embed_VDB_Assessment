import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

BASE_DIR = Path(__file__).resolve().parent.parent

QDRANT_DIR = BASE_DIR / "data" / "T3" / "qdrant_db"
EVAL_FILE = BASE_DIR / "evaluation" / "eval_queries.json"

COLLECTION_NAME = "policy_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(path=str(QDRANT_DIR))

with open(EVAL_FILE, "r", encoding="utf-8") as file:
    evaluation_data = json.load(file)


def evaluate(hnsw_ef):
    hits = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = []

    for item in evaluation_data:
        vector = model.encode(
            item["query"],
            normalize_embeddings=True
        ).tolist()

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=5,
            search_params=models.SearchParams(hnsw_ef=hnsw_ef),
            with_payload=True
        )

        expected_source = item["expected_source"]
        expected_page = item["expected_page"]

        relevant_rank = None

        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}

            if (
                payload.get("source") == expected_source
                and payload.get("page") == expected_page
            ):
                relevant_rank = rank
                break

        for k in [1, 3, 5]:
            if relevant_rank is not None and relevant_rank <= k:
                hits[k] += 1

        reciprocal_ranks.append(
            1 / relevant_rank if relevant_rank else 0
        )

    total = len(evaluation_data)

    return {
        "Hit@1": round(hits[1] / total, 4),
        "Hit@3": round(hits[3] / total, 4),
        "Hit@5": round(hits[5] / total, 4),
        "MRR": round(sum(reciprocal_ranks) / total, 4)
    }


print("\nCONFIGURATION 1: HNSW EF = 32")
result_32 = evaluate(32)
print(result_32)

print("\nCONFIGURATION 2: HNSW EF = 128")
result_128 = evaluate(128)
print(result_128)

print("\nEVALUATION COMPLETED!")

client.close()
del client