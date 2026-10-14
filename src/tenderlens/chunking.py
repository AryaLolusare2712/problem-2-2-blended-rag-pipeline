"""Document loading and overlap-aware chunking."""
from __future__ import annotations
from pathlib import Path
import re
from typing import Iterable

SUPPORTED = {".txt", ".md", ".pdf"}

def load_documents(root: str | Path) -> list[dict]:
    """Load supported documents recursively, preserving source and page metadata."""
    root = Path(root)
    if root.is_file():
        paths = [root] if root.suffix.lower() in SUPPORTED else []
    else:
        paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    docs: list[dict] = []
    for path in paths:
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("Install pypdf to ingest PDF files.") from exc
            reader = PdfReader(str(path))
            for page_no, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    docs.append({"text": text, "source": str(path), "page": page_no})
        else:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                docs.append({"text": text, "source": str(path), "page": None})
    return docs

def chunk_text(text: str, chunk_size: int = 180, overlap: int = 35) -> list[str]:
    """Split into word-based chunks. Size and overlap are measured in words."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    words = re.findall(r"\S+", text)
    if not words:
        return []
    step = chunk_size - overlap
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), step)]

def make_chunks(documents: Iterable[dict], chunk_size: int = 180, overlap: int = 35) -> list[dict]:
    chunks = []
    for doc in documents:
        for idx, content in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            chunks.append({
                "chunk_id": f'{Path(doc["source"]).name}::p{doc.get("page") or 0}::c{idx}',
                "text": content,
                "source": doc["source"],
                "page": doc.get("page"),
                "chunk_index": idx,
            })
    return chunks
