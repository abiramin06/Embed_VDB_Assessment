import json
import re
import tiktoken
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_FILE = Path("data/T2/extracted_documents.jsonl")

documents = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        documents.append(json.loads(line))

print("Total extracted pages:", len(documents))

encoding = tiktoken.get_encoding("cl100k_base")

def fixed_chunks(text, chunk_size=512, overlap=50):

    tokens = encoding.encode(text)
    chunks = []

    start = 0

    while start < len(tokens):

        end = min(start + chunk_size, len(tokens))

        chunk_text = encoding.decode(tokens[start:end])

        chunks.append(chunk_text)

        if end == len(tokens):
            break

        start = end - overlap

    return chunks

def sentence_chunks(text, max_tokens=512):

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:

        if not sentence.strip():
            continue

        sentence_tokens = len(encoding.encode(sentence))

        if current_tokens + sentence_tokens > max_tokens and current_chunk:

            chunks.append(" ".join(current_chunk))

            current_chunk = []
            current_tokens = 0

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

fixed_results = []
sentence_results = []
recursive_results = []

for document in documents:

    text = document["text"]

    fixed_results.extend(
        fixed_chunks(text, 512, 50)
    )

    sentence_results.extend(
        sentence_chunks(text, 512)
    )

    recursive_results.extend(
        recursive_splitter.split_text(text)
    )

print("\n========== CHUNKING COMPARISON ==========")

print("Fixed token chunks:", len(fixed_results))
print("Sentence-based chunks:", len(sentence_results))
print("Recursive chunks:", len(recursive_results))

print("\n========== FIXED CHUNK DETAILS ==========")

for i, chunk in enumerate(fixed_results[:3]):

    print(f"\nChunk {i + 1}")
    print("Token count:", len(encoding.encode(chunk)))
    print("Text:", chunk[:200])

print("\n========== SENTENCE BOUNDARY EXAMPLE ==========")

for i in range(len(fixed_results) - 1):

    current = fixed_results[i]
    following = fixed_results[i + 1]

    if current and following:

        print("Previous chunk ending:")
        print(current[-150:])

        print("\nNext chunk beginning:")
        print(following[:150])

        print("\nFixed chunking may split a sentence.")
        break
print("\nTASK 2 CHUNKING METHODS COMPLETED")