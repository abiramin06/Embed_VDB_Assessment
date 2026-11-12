import sys
import numpy as np
from pathlib import Path
import src.T4_search_api as search_api
import pytest
from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.T4_search_api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    
def test_search_without_api_key():
    response = client.post(
        "/search",
        json={
            "query": "annual report",
            "top_k": 5
        }
    )

    assert response.status_code == 422
    
def test_search_with_invalid_api_key():
    response = client.post(
        "/search",
        headers={"x-api-key": "wrong12345"},
        json={
            "query": "annual report",
            "top_k": 5
        }
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"

def test_search_with_valid_api_key():
    response = client.post(
        "/search",
        headers={"x-api-key": "abirami12345"},
        json={
            "query": "annual report",
            "top_k": 5
        }
    )

    assert response.status_code == 200
    assert "results" in response.json()
    assert "total_results" in response.json()

def test_search_with_empty_query():
    response = client.post(
        "/search",
        headers={"x-api-key": "abirami12345"},
        json={
            "query": "",
            "top_k": 5
        }
    )

    assert response.status_code == 422
def test_upload_without_api_key():
    response = client.post(
        "/documents",
        files={
            "file": (
                "test.txt",
                b"This is a test document for API testing.",
                "text/plain"
            )
        }
    )

    assert response.status_code == 422
def test_successful_document_upload(monkeypatch):
    monkeypatch.setattr(
        search_api,
        "extract_document",
        lambda file_bytes, filename: [
            {"text": "This is sample document content.", "page": 1}
        ]
    )

    monkeypatch.setattr(
        search_api,
        "index_document",
        lambda extracted_pages, filename: 3
    )

    response = client.post(
        "/documents",
        headers={"x-api-key": "abirami12345"},
        files={
            "file": (
                "sample.txt",
                b"This is sample document content.",
                "text/plain"
            )
        }
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "sample.txt"
    assert response.json()["chunks_added"] == 3

def test_extract_txt_document():
    result = search_api.extract_document(
        b"This is a sample text document.",
        "sample.txt"
    )

    assert len(result) == 1
    assert result[0]["text"] == "This is a sample text document."


def test_extract_unsupported_document():
    with pytest.raises(HTTPException) as exc:
        search_api.extract_document(
            b"Sample content",
            "sample.xlsx"
        )

    assert exc.value.status_code == 400

def test_extract_pdf_document():
    import fitz

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((50, 50), "This is a sample PDF document.")

    pdf_bytes = pdf.tobytes()
    pdf.close()

    result = search_api.extract_document(
        pdf_bytes,
        "sample.pdf"
    )

    assert len(result) == 1
    assert "sample PDF document" in result[0]["text"]
    assert result[0]["page"] == 1
def test_extract_docx_document():
    from docx import Document
    from io import BytesIO

    doc = Document()
    doc.add_paragraph("This is a sample DOCX document.")

    buffer = BytesIO()
    doc.save(buffer)

    result = search_api.extract_document(
        buffer.getvalue(),
        "sample.docx"
    )

    assert len(result) == 1
    assert "sample DOCX document" in result[0]["text"]
def test_index_document(monkeypatch):
    captured = {}

def fake_encode(texts, normalize_embeddings=True):
    return [np.array([0.1] * 384) for _ in texts]

    def fake_upsert(collection_name, points):
        captured["collection_name"] = collection_name
        captured["points"] = points

        monkeypatch.setattr(search_api.model, "encode", fake_encode)
        monkeypatch.setattr(search_api.client, "upsert", fake_upsert)

        extracted_pages = [
            {
                "text": "This is a sample document for testing indexing.",
                "page": 1
            }
        ]
        result = search_api.index_document(
            extracted_pages,
            "sample.txt"
        )
        assert result >= 1
        assert captured["collection_name"] == search_api.COLLECTION_NAME
        assert len(captured["points"]) == result
        assert captured["points"][0].payload["source"] == "sample.txt"
def test_search_min_score_filter(monkeypatch):
    from types import SimpleNamespace
    import numpy as np

    def fake_encode(query, normalize_embeddings=True):
        return np.array([0.1] * 384)

    def fake_query_points(**kwargs):
        return SimpleNamespace(points=[
            SimpleNamespace(
                id="1",
                payload={
                    "text": "High score document",
                    "source": "sample.pdf",
                    "page": 1,
                    "section_heading": "Introduction",
                    "doc_type": "pdf"
                },
                score=0.95
            ),
            SimpleNamespace(
                id="2",
                payload={
                    "text": "Low score document",
                    "source": "sample.pdf",
                    "page": 2,
                    "section_heading": "Conclusion",
                    "doc_type": "pdf"
                },
                score=0.40
            )
        ])

    monkeypatch.setattr(search_api.model, "encode", fake_encode)
    monkeypatch.setattr(search_api.client, "query_points", fake_query_points)

    response = client.post(
        "/search",
        headers={"X-API-Key": "abirami12345"},
        json={
            "query": "sample document",
            "top_k": 5,
            "min_score": 0.8
        }
    )

    assert response.status_code == 200
    assert len(response.json()["results"]) == 1
    assert response.json()["results"][0]["score"] == 0.95
def test_search_with_doc_type_filter(monkeypatch):
    from types import SimpleNamespace
    import numpy as np

    captured = {}

    def fake_encode(query, normalize_embeddings=True):
        return np.array([0.1] * 384)

    def fake_query_points(**kwargs):
        captured["query_filter"] = kwargs.get("query_filter")
        return SimpleNamespace(points=[])

    monkeypatch.setattr(search_api.model, "encode", fake_encode)
    monkeypatch.setattr(search_api.client, "query_points", fake_query_points)

    response = client.post(
        "/search",
        headers={"X-API-Key": "abirami12345"},
        json={
            "query": "annual report",
            "top_k": 5,
            "doc_type": "pdf"
        }
    )

    assert response.status_code == 200
    assert captured["query_filter"] is not None