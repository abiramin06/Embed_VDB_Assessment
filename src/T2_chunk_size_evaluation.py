import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "T2" / "extracted_documents.jsonl"

CHUNK_SIZES = [128, 256, 512, 1024]

CHUNK_OVERLAP = 50

# Load extracted documents
documents = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        documents.append(json.loads(line))

print("Total documents:", len(documents))
print("Chunk sizes:", CHUNK_SIZES)

print("\n========== CHUNK SIZE COMPARISON ==========")

for size in CHUNK_SIZES:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=min(CHUNK_OVERLAP, size // 4)
    )

    all_chunks = []

    for document in documents:
        chunks = splitter.split_text(document["text"])
        all_chunks.extend(chunks)

    total_chars = sum(len(chunk) for chunk in all_chunks)

    average_length = (
        total_chars / len(all_chunks)
        if all_chunks else 0
    )

    print(f"\nChunk size: {size}")
    print("Total chunks:", len(all_chunks))
    print("Average chunk length:", round(average_length, 2))
print("\n========== SAMPLE DOCUMENT CONTENT ==========")

for index, document in enumerate(documents[:10], start=1):
    print(f"\nDocument {index}")
    print("Source:", document["source"])
    print("Page:", document["page"])
    print("Text:", document["text"][:500])

# Evaluation questions with known source and page
evaluation_questions = [
    {
        "query": "When was NITI Aayog founded?",
        "expected_source": "Annual-Report-English.pdf",
        "expected_page": 10
    },
    {
        "query": "Who serves as the chairman of NITI Aayog?",
        "expected_source": "Annual-Report-English.pdf",
        "expected_page": 10
    },
    {
        "query": "What are the two main hubs of NITI Aayog?",
        "expected_source": "Annual-Report-English.pdf",
        "expected_page": 11
    },
    {
        "query": "What is the responsibility of Team India Hub?",
        "expected_source": "Annual-Report-English.pdf",
        "expected_page": 11
    },
    {
        "query": "What is the role of Knowledge and Innovation Hub?",
        "expected_source": "Annual-Report-English.pdf",
        "expected_page": 11
    },
    {
        "query": "How many consultants were engaged by NITI Aayog?",
        "expected_source": "Annual-Report-English.pdf",
        "expected_page": 12
    },
    {
        "query": "When was the Development Monitoring and Evaluation Office constituted?",
        "expected_source": "Annual-Report-English.pdf",
        "expected_page": 13
    },
    {
        "query": "Who is the CEO of NITI Aayog according to the organisation structure?",
        "expected_source": "Annual-Report-English.pdf",
        "expected_page": 8
    },
    {
        "query": "Which sectors are included in NITI Aayog's objectives and achievements?",
        "expected_source": "Annual-Report-English.pdf",
        "expected_page": 6
    },
    {
        "query": "Who are the advisers listed in NITI Aayog's organisation structure?",
        "expected_source": "Annual-Report-English.pdf",
        "expected_page": 9
    }
]

print("\nTotal evaluation questions:", len(evaluation_questions))

OUTPUT_EVAL = BASE_DIR / "evaluation" / "chunk_size_eval.json"

OUTPUT_EVAL.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_EVAL, "w", encoding="utf-8") as file:
    json.dump(evaluation_questions, file, indent=4)

print("\nEvaluation questions saved:", OUTPUT_EVAL)

# Load embedding model
print("\nLoading embedding model...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Embedding model loaded successfully!")

# Load evaluation questions
with open(OUTPUT_EVAL, "r", encoding="utf-8") as file:
    eval_data = json.load(file)

print("\n========== RETRIEVAL HIT RATE COMPARISON ==========")

results =[]

for size in CHUNK_SIZES:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=min(CHUNK_OVERLAP, size // 4)
    )

    chunks = []

    for document in documents:
        for chunk_text in splitter.split_text(document["text"]):
            chunks.append({
                "text": chunk_text,
                "source": document["source"],
                "page": document["page"]
            })

    chunk_texts = [chunk["text"] for chunk in chunks]

    chunk_embeddings = model.encode(
        chunk_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    hits = 0

    for item in eval_data:

        query_embedding = model.encode(
            item["query"],
            normalize_embeddings=True
        )

        scores = cosine_similarity(
            [query_embedding],
            chunk_embeddings
        )[0]

        top_indices = np.argsort(scores)[::-1][:5]

        found = any(
            chunks[index]["source"] == item["expected_source"]
            and chunks[index]["page"] == item["expected_page"]
            for index in top_indices
        )

        if found:
            hits += 1

    hit_rate = (hits / len(eval_data)) * 100

    print(f"\nChunk size: {size}")
    print(f"Hits: {hits}/{len(eval_data)}")
    print(f"Hit rate: {hit_rate:.2f}%")
    
results.append({
        "chunk_size": size,
        "hits": hits,
        "total_questions": len(eval_data),
        "hit_rate": round(hit_rate, 2)
    })
# Save comparison results
RESULT_FILE = BASE_DIR / "evaluation" / "chunk_size_results.json"

with open(RESULT_FILE, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nResults saved successfully!")
print("File:", RESULT_FILE)