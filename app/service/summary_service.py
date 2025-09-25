"""Summary generation service."""

import io
import os
import re
import sys
import time
from typing import List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import requests
from openai import OpenAI
from pypdf import PdfReader

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ..models import (
        SummaryConfig, 
        SummaryResult, 
        DocumentType,
        DocumentChunk,
        DocumentSummary
    )
    from .qdrant_service import QdrantService
except ImportError:
    # Fallback for direct execution
    from models.models import (
        SummaryConfig, 
        SummaryResult, 
        DocumentType,
        DocumentChunk,
        DocumentSummary
    )
    from .qdrant_service import QdrantService


class SummaryGenerationError(Exception):
    """Exception raised during summary generation."""
    pass


@dataclass
class DocumentProcessor:
    """Handles document processing and text extraction."""
    
    @staticmethod
    def _read_bytes_from_http(url: str, timeout_sec: int = 60) -> bytes:
        """Read bytes from HTTP URL."""
        response = requests.get(url, timeout=timeout_sec)
        response.raise_for_status()
        return response.content
    
    @staticmethod
    def _read_bytes_from_s3(url: str) -> bytes:
        """Read bytes from S3 URL."""
        try:
            import boto3
        except ImportError:
            raise SummaryGenerationError(
                "boto3 not installed. Add boto3 to requirements or install it to read s3 URLs."
            )
        
        s3_pattern = re.compile(r"^s3://(?P<bucket>[^/]+)/(?P<key>.+)$")
        match = s3_pattern.match(url)
        if not match:
            raise SummaryGenerationError(f"Invalid s3 URL: {url}")
        
        bucket = match.group("bucket")
        key = match.group("key")
        session = boto3.session.Session()
        s3 = session.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    
    @staticmethod
    def _read_bytes_from_gcs(url: str) -> bytes:
        """Read bytes from Google Cloud Storage URL."""
        try:
            from google.cloud import storage
        except ImportError:
            raise SummaryGenerationError(
                "google-cloud-storage not installed. Install it to read gs URLs."
            )
        
        gcs_pattern = re.compile(r"^gs://(?P<bucket>[^/]+)/(?P<blob>.+)$")
        match = gcs_pattern.match(url)
        if not match:
            raise SummaryGenerationError(f"Invalid gs URL: {url}")
        
        bucket_name = match.group("bucket")
        blob_name = match.group("blob")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_bytes()
    
    @staticmethod
    def fetch_bytes(url: str) -> bytes:
        """Fetch bytes from various URL schemes."""
        if url.startswith("s3://"):
            return DocumentProcessor._read_bytes_from_s3(url)
        if url.startswith("gs://"):
            return DocumentProcessor._read_bytes_from_gcs(url)
        if url.startswith("http://") or url.startswith("https://"):
            return DocumentProcessor._read_bytes_from_http(url)
        
        # Local path fallback
        if os.path.exists(url):
            with open(url, "rb") as f:
                return f.read()
        
        raise SummaryGenerationError(
            "Unsupported URL scheme. Use http(s)://, s3://, gs://, or a local file path."
        )
    
    @staticmethod
    def detect_document_type(data: bytes, url: str) -> DocumentType:
        """Detect document type from data and URL."""
        if data[:5] == b"%PDF-" or url.lower().endswith(".pdf"):
            return DocumentType.PDF
        if url.lower().endswith(".txt"):
            return DocumentType.TEXT
        
        # Heuristic: if decodable as utf-8 and contains mostly text
        try:
            text = data.decode("utf-8")
            control_ratio = sum(1 for c in text if ord(c) < 9 or (13 < ord(c) < 32)) / max(1, len(text))
            if control_ratio < 0.01:
                return DocumentType.TEXT
        except Exception:
            pass
        
        return DocumentType.UNKNOWN
    
    @staticmethod
    def extract_text_from_pdf(data: bytes) -> str:
        """Extract text from PDF data."""
        reader = PdfReader(io.BytesIO(data))
        pages: List[str] = []
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            pages.append(page_text)
        return "\n\n".join(pages).strip()
    
    @staticmethod
    def extract_text(data: bytes, url: str) -> str:
        """Extract text from document data."""
        doc_type = DocumentProcessor.detect_document_type(data, url)
        
        if doc_type == DocumentType.PDF:
            return DocumentProcessor.extract_text_from_pdf(data)
        elif doc_type == DocumentType.TEXT:
            try:
                return data.decode("utf-8", errors="ignore")
            except Exception as exc:
                raise SummaryGenerationError(f"Failed to decode text: {exc}")
        
        raise SummaryGenerationError(
            "Unsupported file type. Only PDF and UTF-8 text are supported currently."
        )


class SummaryService:
    """Service for generating document summaries."""
    
    def __init__(self, config: Optional[SummaryConfig] = None, qdrant_service: Optional[QdrantService] = None):
        """Initialize summary service."""
        self.config = config or SummaryConfig()
        self.qdrant_service = qdrant_service
        self._openai_client = None
    
    @property
    def openai_client(self) -> OpenAI:
        """Get OpenAI client (lazy initialization)."""
        if self._openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise SummaryGenerationError(
                    "OPENAI_API_KEY is not set. Please set it in the environment."
                )
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text into manageable pieces."""
        if len(text) <= self.config.max_input_chars_per_chunk:
            return [text]
        
        chunks: List[str] = []
        start = 0
        end = len(text)
        
        while start < end:
            chunk_end = min(end, start + self.config.max_input_chars_per_chunk)
            chunk = text[start:chunk_end]
            chunks.append(chunk)
            
            if chunk_end >= end:
                break
            
            start = chunk_end - self.config.overlap_chars
            if start < 0:
                start = 0
        
        return chunks
    
    def _map_summarize(self, chunk: str) -> str:
        """Summarize a single chunk."""
        system_prompt = (
            "You are a senior analyst and expert narrator. Write vivid, structured summaries "
            "optimized for spoken narration. Use concise sentences, logical flow, and clear "
            "headings with short bullet points where helpful. Avoid redundancy."
        )
        user_prompt = (
            "Summarize the following content for narration. Capture key ideas, decisions, "
            "data points, and any processes described. Provide a strong narrative arc, a "
            "one-sentence high-level overview, and callouts for critical quotes or figures.\n\n"
            f"CONTENT:\n{chunk}"
        )
        
        completion = self.openai_client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (completion.choices[0].message.content or "").strip()
    
    def _reduce_summaries(self, partial_summaries: List[str]) -> str:
        """Merge partial summaries into a single summary."""
        system_prompt = (
            "You merge partial narration summaries into a single, cohesive narrative. "
            "Preserve fidelity to the source, remove redundancy, and maintain a clear "
            "beginning, middle, and end. Optimize for voice-over narration."
        )
        joined = "\n\n---\n\n".join(partial_summaries)
        user_prompt = (
            "Merge the following partial summaries into one comprehensive narration-ready "
            "summary with: (1) high-level overview, (2) thematic sections with clear "
            "headings, (3) concise bullet points for details, (4) conclusion with key "
            "takeaways.\n\nPARTIAL SUMMARIES:\n\n" + joined
        )
        
        completion = self.openai_client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (completion.choices[0].message.content or "").strip()
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        embeddings = self.openai_client.embeddings.create(
            model=self.config.embed_model,
            input=texts,
        )
        return [item.embedding for item in embeddings.data]
    
    def generate_summary(self, url: str, doc_id: Optional[str] = None) -> SummaryResult:
        """Generate summary for a document."""
        start_time = time.time()
        
        # Extract text from document
        data = DocumentProcessor.fetch_bytes(url)
        text = DocumentProcessor.extract_text(data, url)
        
        if not text.strip():
            raise SummaryGenerationError("No extractable text found in the provided document.")
        
        # Chunk text
        chunks = self._chunk_text(text)
        
        # Map step: summarize each chunk
        partials: List[str] = []
        for chunk in chunks:
            partials.append(self._map_summarize(chunk))
        
        # Reduce step: iteratively merge until one remains or max passes reached
        passes = 0
        while len(partials) > 1 and passes < self.config.max_reduce_passes:
            merged: List[str] = []
            for i in range(0, len(partials), 4):
                group = partials[i : i + 4]
                merged.append(self._reduce_summaries(group))
            partials = merged
            passes += 1
        
        final_summary = partials[0] if partials else ""
        if not final_summary:
            raise SummaryGenerationError("Failed to generate summary.")
        
        # Store in vector database if Qdrant service is available
        if self.qdrant_service:
            try:
                self._store_in_vector_db(chunks, final_summary, url, doc_id)
            except Exception as e:
                print(f"Error storing in vector database: {e}")
        
        processing_time = time.time() - start_time
        
        return SummaryResult(
            summary=final_summary,
            chunks=chunks,
            doc_id=doc_id or os.path.basename(url) or "unknown",
            source_url=url,
            processing_time=processing_time
        )
    
    def _store_in_vector_db(self, chunks: List[str], summary: str, url: str, doc_id: Optional[str]) -> None:
        """Store chunks and summary in vector database."""
        if not self.qdrant_service:
            return
        
        # Generate embeddings
        texts_to_embed = chunks + [summary]
        vectors = self._generate_embeddings(texts_to_embed)
        
        # Prepare payloads and IDs
        payloads = []
        ids = []
        final_doc_id = doc_id or os.path.basename(url) or "unknown"
        
        # Add chunk payloads
        for idx, content in enumerate(chunks):
            payloads.append({
                "type": "chunk",
                "doc_id": final_doc_id,
                "chunk_index": idx,
                "source_url": url,
                "text": content,
            })
            ids.append(idx)
        
        # Add summary payload
        payloads.append({
            "type": "summary",
            "doc_id": final_doc_id,
            "chunk_index": -1,
            "source_url": url,
            "text": summary,
        })
        ids.append(len(chunks))
        
        # Upsert to Qdrant
        self.qdrant_service.upsert_embeddings(vectors, payloads, ids)
