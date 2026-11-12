import json
import hashlib
from pathlib import Path
from datetime import date
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_FILE = Path("data/T2/extracted_documents.jsonl")
OUTPUT_FILE = Path("data/T2/metadata_chunks.jsonl")

# Load documents
documents = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        documents.append(json.loads(line))


# Recursive chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)


# Generate stable ID
def generate_chunk_id(source, page, text):

    content = f"{source}|{page}|{text.strip()}"

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


# Detect section heading
def get_section_heading(text):

    for line in text.splitlines():

        line = line.strip()

        if line and len(line) < 100:
            return line

    return "General"


# Create metadata
def create_chunks(documents):

    chunks = []

    for document in documents:

        text = document["text"]

        text_chunks = splitter.split_text(text)

        for index, chunk_text in enumerate(text_chunks):

            source = document["source"]
            page = document["page"]

            chunk = {
                "id": generate_chunk_id(
                    source, page, chunk_text
                ),
                "text": chunk_text,
                "metadata": {
                    "source": source,
                    "page": page,
                    "section_heading": get_section_heading(chunk_text),
                    "created_date": date.today().isoformat(),
                    "chunk_index": index,
                    "doc_type": Path(source).suffix.lower().replace(".", "")
                }
            }

            chunks.append(chunk)

    return chunks


# Citation function
def describe_chunk(chunk):

    metadata = chunk["metadata"]

    return (
        f"Source: {metadata['source']} | "
        f"Page: {metadata['page']} | "
        f"Section: {metadata['section_heading']}"
    )


# Simulated vector index using dictionary
def upsert_chunks(index, chunks):

    for chunk in chunks:
        index[chunk["id"]] = chunk

    return index


if __name__ == "__main__":

    print("\n========== CREATING CHUNKS ==========")

    chunks = create_chunks(documents)

    print("Total chunks:", len(chunks))

    # Save metadata chunks
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        for chunk in chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    # Display metadata example
    print("\n========== METADATA EXAMPLE ==========")

    print(json.dumps(
        chunks[0],
        indent=4,
        ensure_ascii=False
    ))

    print("\n========== CITATION EXAMPLE ==========")

    print(describe_chunk(chunks[0]))

    # First indexing
    vector_index = {}

    upsert_chunks(vector_index, chunks)

    count1 = len(vector_index)

    print("\n========== UPSERT TEST ==========")
    print("Run 1:", count1, "points")

    # Second indexing
    upsert_chunks(vector_index, chunks)

    count2 = len(vector_index)

    print("Run 2:", count2, "points")

    print("Count unchanged:", count1 == count2)

    # Edit one paragraph in one chunk
    edited_chunks = [chunk.copy() for chunk in chunks]

    edited_chunks[0] = {
        **chunks[0],
        "text": chunks[0]["text"] + "\nAssessment edit test."
    }

    edited_chunks[0]["metadata"] = {
        **chunks[0]["metadata"]
    }

    edited_chunks[0]["id"] = generate_chunk_id(
        edited_chunks[0]["metadata"]["source"],
        edited_chunks[0]["metadata"]["page"],
        edited_chunks[0]["text"]
    )

    # Re-upsert edited chunk correctly

    old_chunk_id = chunks[0]["id"]
    new_chunk = edited_chunks[0]
    new_chunk_id = new_chunk["id"]

    # Delete old chunk
    vector_index.pop(old_chunk_id, None)

    # Insert updated chunk
    vector_index[new_chunk_id] = new_chunk

    print("\n========== EDIT TEST ==========")
    print("Old chunk ID:", old_chunk_id)
    print("New chunk ID:", new_chunk_id)
    print("Replaced chunks: 1")
    print("New index count:", len(vector_index))
    print("Old index count:", count2)
    print("Count unchanged:", len(vector_index) == count2)    