import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from fastapi import UploadFile, File
from io import BytesIO
import hashlib
import uuid
import fitz
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent

QDRANT_DIR = BASE_DIR / "data" / "T3" / "qdrant_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "policy_documents"

VECTOR_SIZE = 384

API_KEY = os.getenv("SEARCH_API_KEY", "dev-secret-key")

# Initialize FastAPI
app = FastAPI(
    title="Document Semantic Search API",
    description="Semantic search using Qdrant and Sentence Transformers",
    version="1.0.0"
)

# Load embedding model
model = SentenceTransformer(MODEL_NAME)

# Connect to existing Qdrant database
client = QdrantClient(
    path=str(QDRANT_DIR)
)

print("Search API configuration loaded successfully.")

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_type: Optional[str] = None
    min_score: Optional[float] = None

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "message": "Search API is running",
        "collection": COLLECTION_NAME
    }
    
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return True    
@app.post("/search")
def semantic_search(
    request: SearchRequest,
    authenticated: bool = Depends(verify_api_key)):

    query_vector = model.encode(
        request.query,
        normalize_embeddings=True
    ).tolist()

    search_filter = None

    if request.doc_type:
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_type",
                    match=models.MatchValue(value=request.doc_type)
                )
            ]
        )
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=request.top_k,
        with_payload=True
    )

    results = []

    for point in search_result.points:

        if request.min_score is not None:
            if point.score < request.min_score:
                continue

        payload = point.payload or {}

        if request.doc_type:
            if payload.get("doc_type") != request.doc_type:
                continue

        results.append({
            "id": point.id,
            "text": payload.get("text", ""),
            "source": payload.get("source", ""),
            "page": payload.get("page"),
            "section_heading": payload.get("section_heading", ""),
            "doc_type": payload.get("doc_type", ""),
            "score": round(point.score, 4)
        })

    return {
        "query": request.query,
        "total_results": len(results),
        "results": results
    }
def verify_api_key(x_api_key: str = Header(...)):

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return True

def extract_document(file_bytes: bytes, filename: str):
    extension = Path(filename).suffix.lower()
    extracted = []

    if extension == ".pdf":
        pdf = fitz.open(stream=file_bytes, filetype="pdf")

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text()

            if text.strip():
                extracted.append({
                    "text": text,
                    "page": page_number
                })

        pdf.close()

    elif extension == ".docx":
        document = Document(BytesIO(file_bytes))
        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )

        if text.strip():
            extracted.append({
                "text": text,
                "page": None
            })

    elif extension == ".txt":
        text = file_bytes.decode("utf-8", errors="ignore")

        if text.strip():
            extracted.append({
                "text": text,
                "page": None
            })

    else:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOCX and TXT files are supported"
        )

    return extracted
def index_document(extracted_pages, filename):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )

    points = []

    for page_data in extracted_pages:
        chunks = splitter.split_text(page_data["text"])

        for chunk_index, chunk in enumerate(chunks):
            vector = model.encode(
                chunk,
                normalize_embeddings=True
            ).tolist()

            page = page_data["page"]

            unique_text = f"{filename}|{page}|{chunk_index}|{chunk}"
            chunk_hash = hashlib.sha256(
                unique_text.encode("utf-8")
            ).hexdigest()

            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_DNS, chunk_hash)
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source": filename,
                        "page": page,
                        "section_heading": "",
                        "doc_type": Path(filename).suffix.lower().replace(".", ""),
                        "chunk_index": chunk_index
                    }
                )
            )

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    return len(points)
@app.post("/documents")
def upload_document(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key)
):
    filename = Path(file.filename).name

    file_bytes = file.file.read()

    extracted_pages = extract_document(
        file_bytes,
        filename
    )

    if not extracted_pages:
        raise HTTPException(
            status_code=400,
            detail="No readable text found in document"
        )

    chunks_added = index_document(
        extracted_pages,
        filename
    )

    return {
        "message": "Document uploaded and indexed successfully",
        "filename": filename,
        "chunks_added": chunks_added
    }
