"""Qdrant vector database service."""

import os
import sys
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import QdrantConfig, DocumentEmbedding, SearchResult

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantService:
    """Service for managing Qdrant vector database operations."""
    
    def __init__(self, config: Optional[QdrantConfig] = None):
        """Initialize Qdrant service with configuration."""
        self.config = config or self._get_default_config()
        print(self.config)
        self.client = self._create_client()
    
    def _get_default_config(self) -> QdrantConfig:
        """Get default configuration from environment variables."""
        return QdrantConfig(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv("QDRANT_COLLECTION", "dev-collection"),
            vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", "3072")),
        )
    
    def _create_client(self) -> QdrantClient:
        """Create Qdrant client based on configuration."""
        if self.config.url:
            logger.info("Creating QdrantClient with remote URL: %s", self.config.url)
            return QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=30,
                check_compatibility=False
            )
        else:
            logger.info("Creating QdrantClient with local host")
            return QdrantClient(
                host=os.getenv("QDRANT_HOST", "127.0.0.1"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
                prefer_grpc=False,
                check_compatibility=False,
            )
    
    def ensure_collection(self) -> None:
        """
        Ensure collection exists. Skip if management APIs are disabled (cloud).
        """
        try:
            existing = self.client.get_collections()
            names = {c.name for c in existing.collections}
            if self.config.collection_name in names:
                logger.debug("Collection %s already exists", self.config.collection_name)
                return

            logger.info(
                "Creating collection %s (size=%d)", 
                self.config.collection_name, 
                self.config.vector_size
            )
            
            distance = getattr(qmodels.Distance, self.config.distance.upper())
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.config.vector_size,
                    distance=distance,
                    on_disk=self.config.on_disk,
                ),
            )
        except UnexpectedResponse as e:
            msg = str(e).lower()
            if "403" in msg or "forbidden" in msg:
                logger.warning(
                    "Received 403/forbidden while checking/creating collections. "
                    "This usually means you're on Qdrant Cloud and management APIs are disabled. "
                    "Skipping collection creation. Make sure the collection exists in the Qdrant dashboard."
                )
                return
            raise
        except Exception:
            logger.exception("Unexpected error while ensuring collection")
            raise
    
    def upsert_embeddings(
        self, 
        vectors: List[List[float]], 
        payloads: List[Dict[str, Any]], 
        ids: Optional[List[str]] = None
    ) -> None:
        """Upsert embeddings into the collection."""
        self.ensure_collection()
        
        try:
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=qmodels.Batch(
                    ids=ids,
                    vectors=vectors,
                    payloads=payloads,
                ),
            )
            logger.info(
                "Upserted %d points into collection %s", 
                len(vectors), 
                self.config.collection_name
            )
        except UnexpectedResponse as e:
            msg = str(e).lower()
            if "403" in msg or "forbidden" in msg:
                raise PermissionError(
                    "Qdrant returned 403 (forbidden) while upserting. "
                    "This usually means your API key does not have 'write' permissions for this cluster/collection. "
                    "Check the Qdrant Cloud dashboard and API-key scopes."
                ) from e
            raise
        except Exception:
            logger.exception("Failed to upsert points")
            raise
    
    def search(
        self,
        vector: List[float],
        limit: int = 10,
        query_filter: Optional[qmodels.Filter] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        try:
            self.ensure_collection()
        except Exception:
            logger.debug("Proceeding to search despite ensure_collection issues")

        try:
            results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )
            
            return [
                SearchResult(
                    id=str(result.id),
                    score=result.score,
                    payload=result.payload or {}
                )
                for result in results
            ]
        except UnexpectedResponse as e:
            msg = str(e).lower()
            if "403" in msg or "forbidden" in msg:
                raise PermissionError(
                    "Qdrant returned 403 (forbidden) while searching. "
                    "Your API key may be missing read/search permissions."
                ) from e
            raise
    
    def delete_points(self, ids: List[str]) -> None:
        """Delete points by IDs."""
        try:
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=qmodels.PointIdsList(points=ids)
            )
            logger.info("Deleted %d points from collection %s", len(ids), self.config.collection_name)
        except Exception:
            logger.exception("Failed to delete points")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        try:
            collection = self.client.get_collection(self.config.collection_name)
            return {
                "name": collection.config.params.vectors.size,
                "vector_size": collection.config.params.vectors.size,
                "distance": collection.config.params.vectors.distance,
                "points_count": collection.points_count,
            }
        except Exception:
            logger.exception("Failed to get collection info")
            return {}
