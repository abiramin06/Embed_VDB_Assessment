import json
import random
from pathlib import Path

from qdrant_client import QdrantClient

BASE_DIR = Path(__file__).resolve().parent.parent

QDRANT_DIR = BASE_DIR / "data" / "T3" / "qdrant_db"
OUTPUT_FILE = BASE_DIR / "evaluation" / "eval_queries.json"

COLLECTION_NAME = "policy_documents"

client = QdrantClient(path=str(QDRANT_DIR))

points, _ = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=10000,
    with_payload=True,
    with_vectors=False
)

valid_points = []

for point in points:
    payload = point.payload or {}
    text = payload.get("text", "").strip()

    if len(text.split()) >= 20:
        valid_points.append(payload)

random.seed(42)
selected = random.sample(valid_points, min(15, len(valid_points)))

evaluation_data = []

for index, payload in enumerate(selected, start=1):
    words = payload["text"].split()

    evaluation_data.append({
        "id": index,
        "query": " ".join(words[:12]),
        "expected_source": payload.get("source"),
        "expected_page": payload.get("page")
    })

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(evaluation_data, file, indent=4, ensure_ascii=False)

print("Evaluation dataset created from Qdrant!")
print("Available records:", len(valid_points))
print("Total queries:", len(evaluation_data))

client.close()
del client