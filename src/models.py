"""Data models for video narration generation."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    TEXT = "text"
    UNKNOWN = "unknown"


class SummaryConfig(BaseModel):
    """Configuration for summary generation."""
    model: str = "gpt-4o-mini"
    max_input_chars_per_chunk: int = 12000
    overlap_chars: int = 800
    max_reduce_passes: int = 3
    temperature: float = 0.2
    embed_model: str = "text-embedding-3-large"
    qdrant_collection: str = "documents"


class QdrantConfig(BaseModel):
    """Configuration for Qdrant vector database."""
    url: Optional[str] = None
    api_key: Optional[str] = None
    collection_name: str = "dev-collection"
    vector_size: int = 3072
    distance: str = "cosine"
    on_disk: bool = True


class DocumentChunk(BaseModel):
    """Represents a chunk of a document."""
    chunk_index: int
    text: str
    doc_id: str
    source_url: str
    chunk_type: str = "chunk"


class DocumentSummary(BaseModel):
    """Represents a document summary."""
    text: str
    doc_id: str
    source_url: str
    chunk_type: str = "summary"
    chunk_index: int = -1


class DocumentEmbedding(BaseModel):
    """Represents an embedding with metadata."""
    vector: List[float]
    payload: Dict[str, Any]
    id: str


class SummaryResult(BaseModel):
    """Result of summary generation."""
    summary: str
    chunks: List[str]
    doc_id: str
    source_url: str
    processing_time: float


class SearchResult(BaseModel):
    """Result of vector search."""
    id: str
    score: float
    payload: Dict[str, Any]


class Summary(BaseModel):
    title: str
    content: str
    estimated_duration: int = 5
    attendee: str = None


class Slide(BaseModel):
    slide_number: int
    narration_text: str
    slide_json: dict = Field(default_factory=dict)


class NarrationResult(BaseModel):
    title: str
    slides: List[Slide]
    created_at: datetime = Field(default_factory=datetime.utcnow)
