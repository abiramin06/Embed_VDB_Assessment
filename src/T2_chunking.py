import fitz
import json
from pathlib import Path
from docx import Document

# Document folder
DATA_DIR = Path("data/T2/documents")

# Output file
OUTPUT_FILE = Path("data/T2/extracted_documents.jsonl")


def extract_pdf(file_path):
    documents = []

    pdf = fitz.open(file_path)

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text("text")

        if text.strip():
            documents.append({
                "source": file_path.name,
                "page": page_number,
                "text": text
            })

    pdf.close()
    return documents


def extract_docx(file_path):
    doc = Document(file_path)

    text = "\n".join(
        paragraph.text
        for paragraph in doc.paragraphs
        if paragraph.text.strip()
    )

    return [{
        "source": file_path.name,
        "page": 1,
        "text": text
    }]


def extract_txt(file_path):
    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    return [{
        "source": file_path.name,
        "page": 1,
        "text": text
    }]


def extract_all_documents():

    all_documents = []

    for file_path in DATA_DIR.iterdir():

        if file_path.suffix.lower() == ".pdf":
            extracted = extract_pdf(file_path)

        elif file_path.suffix.lower() == ".docx":
            extracted = extract_docx(file_path)

        elif file_path.suffix.lower() == ".txt":
            extracted = extract_txt(file_path)

        else:
            continue

        all_documents.extend(extracted)

        print(
            f"Extracted: {file_path.name} "
            f"({len(extracted)} pages/sections)"
        )

    return all_documents


if __name__ == "__main__":

    documents = extract_all_documents()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")

    print("\n========== EXTRACTION SUMMARY ==========")
    print("Total documents/pages:", len(documents))
    print("Output saved:", OUTPUT_FILE)
    print("Document extraction completed!")