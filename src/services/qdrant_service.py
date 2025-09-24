# qdrant_service.py
import os
import logging
from dataclasses import dataclass
from typing import List, Optional
import dotenv

dotenv.load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QdrantConfig:
    url: Optional[str] = None
    api_key: Optional[str] = None
    collection_name: str = "dev-collection"
    vector_size: int = 3072
    distance: qmodels.Distance = qmodels.Distance.COSINE
    on_disk: bool = True


def get_qdrant_client(cfg: Optional[QdrantConfig] = None) -> QdrantClient:
    # Merge env vars with provided cfg (env wins if cfg fields are None)
    if cfg is None:
        cfg = QdrantConfig(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv("QDRANT_COLLECTION", "dev-collection"),
            vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", "3072")),
        )
    else:
        if cfg.url is None:
            cfg.url = os.getenv("QDRANT_URL") or cfg.url
        if cfg.api_key is None:
            cfg.api_key = os.getenv("QDRANT_API_KEY") or cfg.api_key
        # allow overriding collection name/vector size by env
        cfg.collection_name = os.getenv("QDRANT_COLLECTION", cfg.collection_name)
        cfg.vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", str(cfg.vector_size)))

    # Use remote URL if provided (Cloud/managed), else fallback to local host
    if cfg.url:
        logger.info("Creating QdrantClient remote url=%s", cfg.url)
        # disable the version compatibility check (common for managed clusters)
        return QdrantClient(
            url=cfg.url, api_key=cfg.api_key, timeout=30, check_compatibility=False
        )
    else:
        logger.info("Creating QdrantClient local host")
        return QdrantClient(
            host=os.getenv("QDRANT_HOST", "127.0.0.1"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            prefer_grpc=False,
            check_compatibility=False,
        )


def ensure_collection(client: QdrantClient, cfg: QdrantConfig) -> None:
    """
    Try to check/create collection. If the server returns 403 (managed cloud),
    we log and skip creation because many managed clusters disallow management APIs.
    """
    try:
        existing = client.get_collections()
        names = {c.name for c in existing.collections}
        if cfg.collection_name in names:
            logger.debug("Collection %s already exists", cfg.collection_name)
            return

        logger.info(
            "Creating collection %s (size=%d)", cfg.collection_name, cfg.vector_size
        )
        client.create_collection(
            collection_name=cfg.collection_name,
            vectors_config=qmodels.VectorParams(
                size=cfg.vector_size,
                distance=cfg.distance,
                on_disk=cfg.on_disk,
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
        # re-raise other unexpected responses
        raise
    except Exception:
        logger.exception("Unexpected error while ensuring collection")
        raise


def upsert_points(
    client: QdrantClient,
    cfg: QdrantConfig,
    vectors: List[List[float]],
    payloads: List[dict],
    ids: Optional[List[str]] = None,
) -> None:
    # Attempt to ensure collection (the function itself will skip if forbidden)
    ensure_collection(client, cfg)

    try:
        client.upsert(
            collection_name=cfg.collection_name,
            points=qmodels.Batch(
                ids=ids,
                vectors=vectors,
                payloads=payloads,
            ),
        )
        logger.info(
            "Upserted %d points into collection %s", len(vectors), cfg.collection_name
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
    client: QdrantClient,
    cfg: QdrantConfig,
    vector: List[float],
    limit: int = 10,
    query_filter: Optional[qmodels.Filter] = None,
):
    # ensure_collection may be skipped for cloud but that's OK for searches
    try:
        ensure_collection(client, cfg)
    except Exception:
        # If ensure_collection failed for any reason, proceed to search anyway;
        # many managed clusters allow search even if management APIs are disabled.
        logger.debug("Proceeding to search despite ensure_collection issues")

    try:
        return client.search(
            collection_name=cfg.collection_name,
            query_vector=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )
    except UnexpectedResponse as e:
        msg = str(e).lower()
        if "403" in msg or "forbidden" in msg:
            raise PermissionError(
                "Qdrant returned 403 (forbidden) while searching. "
                "Your API key may be missing read/search permissions."
            ) from e
        raise
