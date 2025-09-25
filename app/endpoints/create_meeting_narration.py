# """Integrated workflow for summary generation and vector storage."""

import os
import sys
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import asdict
import qdrant_client

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.models import Summary
from models.models import SummaryConfig, QdrantConfig
from service.summary_service import SummaryService
from service.qdrant_service import QdrantService
from service.narration_gen_service import NarrationGenerator


class MeetingContentGenerationPipeline:
    """Integrated processor that combines summary generation and vector storage."""

    def __init__(
        self,
        summary_config: Optional[SummaryConfig] = None,
        qdrant_config: Optional[QdrantConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the integrated processor.

        Args:
            summary_config: Configuration for summary generation
            qdrant_config: Configuration for Qdrant vector database
            logger: Optional logger instance. If None, creates a new one.
        """
        self.logger = logger or self._setup_logger()

        try:
            self.summary_config = summary_config or SummaryConfig()
            self.qdrant_config = qdrant_config or QdrantConfig()

            # Initialize services
            self.qdrant_service = QdrantService(self.qdrant_config)
            self.summary_service = SummaryService(
                self.summary_config, self.qdrant_service
            )

            self.logger.info(
                "MeetingContentGenerationPipeline initialized successfully"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to initialize MeetingContentGenerationPipeline: {e}"
            )
            raise

    def _setup_logger(self) -> logging.Logger:
        """Setup logger with appropriate configuration."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Only add handler if logger doesn't have any handlers
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # Set level from environment variable or default to INFO
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))

        return logger

    def process_document(
        self, url: str, doc_id: Optional[str] = None, store_vectors: bool = True
    ) -> Dict[str, Any]:
        """
        Process a document: extract text, generate summary, and store in vector DB.

        Args:
            url: URL or path to the document
            doc_id: Optional document ID
            store_vectors: Whether to store embeddings in vector database

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()

        self.logger.info(f"Starting document processing: {url}")
        self.logger.debug(
            f"Parameters - doc_id: {doc_id}, store_vectors: {store_vectors}"
        )

        try:
            # Validate inputs
            if not url or not isinstance(url, str):
                raise ValueError("URL must be a non-empty string")

            # Generate summary (this will also store in vector DB if enabled)
            original_qdrant_service = None
            if not store_vectors:
                self.logger.debug(
                    "Temporarily disabling Qdrant service for summary-only processing"
                )
                original_qdrant_service = self.summary_service.qdrant_service
                self.summary_service.qdrant_service = None

            result = self.summary_service.generate_summary(url, doc_id)

            # Restore Qdrant service if it was temporarily disabled
            if original_qdrant_service is not None:
                self.summary_service.qdrant_service = original_qdrant_service

            processing_time = time.time() - start_time

            response_data = {
                "success": True,
                "doc_id": result.doc_id,
                "source_url": result.source_url,
                "summary": result.summary,
                "chunks_count": len(result.chunks),
                "processing_time": round(processing_time, 3),
                "summary_length": len(result.summary),
                "stored_in_vector_db": store_vectors,
            }

            self.logger.info(
                f"Document processing completed successfully - "
                f"doc_id: {result.doc_id}, "
                f"chunks: {len(result.chunks)}, "
                f"processing_time: {processing_time:.3f}s"
            )

            return response_data

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to process document {url}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return {
                "success": False,
                "error": str(e),
                "processing_time": round(processing_time, 3),
                "doc_id": doc_id,
                "source_url": url,
            }

    def search_similar_documents(
        self, query_vector: list, limit: int = 10, doc_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            doc_type: Optional filter by document type ('chunk' or 'summary')

        Returns:
            Dictionary with search results
        """
        self.logger.info(
            f"Searching similar documents - limit: {limit}, doc_type: {doc_type}"
        )

        try:
            # Validate inputs
            if not query_vector or not isinstance(query_vector, list):
                raise ValueError("Query vector must be a non-empty list")

            if limit <= 0:
                raise ValueError("Limit must be positive")

            if doc_type and doc_type not in ["chunk", "summary"]:
                raise ValueError("doc_type must be either 'chunk' or 'summary'")

            # Create filter if doc_type is specified
            query_filter = None
            if doc_type:
                try:
                    from qdrant_client.http import models as qmodels

                    query_filter = qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="type", match=qmodels.MatchValue(value=doc_type)
                            )
                        ]
                    )
                    self.logger.debug(f"Created filter for doc_type: {doc_type}")
                except ImportError as e:
                    self.logger.warning(
                        f"Could not import qdrant models for filtering: {e}"
                    )

            results = self.qdrant_service.search(
                vector=query_vector, limit=limit, query_filter=query_filter
            )

            processed_results = [
                {
                    "id": result.id,
                    "score": round(result.score, 4),
                    "type": result.payload.get("type"),
                    "doc_id": result.payload.get("doc_id"),
                    "text_preview": result.payload.get("text", "")[:200] + "...",
                }
                for result in results
            ]

            response_data = {
                "success": True,
                "results_count": len(results),
                "results": processed_results,
            }

            self.logger.info(f"Search completed - found {len(results)} results")
            self.logger.debug(f"Search results: {[r['id'] for r in processed_results]}")

            return response_data

        except Exception as e:
            error_msg = f"Failed to search similar documents: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return {"success": False, "error": str(e)}

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection."""
        self.logger.info("Retrieving collection statistics")

        try:
            info = self.qdrant_service.get_collection_info()

            response_data = {
                "success": True,
                "collection_name": self.qdrant_config.collection_name,
                "vector_size": info.get("vector_size", "unknown"),
                "points_count": info.get("points_count", "unknown"),
                "distance_metric": self.qdrant_config.distance,
            }

            self.logger.info(
                f"Collection stats retrieved - "
                f"collection: {response_data['collection_name']}, "
                f"points: {response_data['points_count']}"
            )

            return response_data

        except Exception as e:
            error_msg = f"Failed to get collection stats: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return {"success": False, "error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services."""
        self.logger.info("Performing health check")

        health_status = {"success": True, "timestamp": time.time(), "services": {}}

        # Check Qdrant service
        try:
            self.qdrant_service.get_collection_info()
            health_status["services"]["qdrant"] = {
                "status": "healthy",
                "collection": self.qdrant_config.collection_name,
            }
            self.logger.debug("Qdrant service health check passed")
        except Exception as e:
            health_status["services"]["qdrant"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["success"] = False
            self.logger.warning(f"Qdrant service health check failed: {e}")

        # Check summary service (basic validation)
        try:
            if self.summary_service:
                health_status["services"]["summary"] = {
                    "status": "healthy",
                    "model": self.summary_config.model,
                }
                self.logger.debug("Summary service health check passed")
            else:
                raise ValueError("Summary service not properly initialized")
        except Exception as e:
            health_status["services"]["summary"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["success"] = False
            self.logger.warning(f"Summary service health check failed: {e}")

        overall_status = "healthy" if health_status["success"] else "unhealthy"
        self.logger.info(f"Health check completed - overall status: {overall_status}")

        return health_status


def setup_logging() -> logging.Logger:
    """Setup application-wide logging configuration."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv(
        "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            # Add file handler if LOG_FILE environment variable is set
            *(
                [logging.FileHandler(os.getenv("LOG_FILE"))]
                if os.getenv("LOG_FILE")
                else []
            ),
        ],
    )

    return logging.getLogger(__name__)


def run_pipeline():
    """Main function demonstrating the integrated workflow."""
    logger = setup_logging()
    logger.info("=== Starting Integrated Document Processing Workflow ===")

    try:
        # Configure services
        summary_config = SummaryConfig(
            model=os.getenv("SUMMARY_MODEL", "gpt-4o-mini"),
            max_input_chars_per_chunk=int(os.getenv("MAX_CHARS_PER_CHUNK", "10000")),
            temperature=float(os.getenv("SUMMARY_TEMPERATURE", "0.2")),
        )

        qdrant_config = QdrantConfig(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv("QDRANT_COLLECTION", "integrated-docs"),
            vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", "3072")),
        )

        # Validate required environment variables
        required_env_vars = ["QDRANT_URL"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

        # Initialize integrated processor
        processor = MeetingContentGenerationPipeline(
            summary_config, qdrant_config, logger
        )

        # Perform health check
        logger.info("Performing initial health check...")
        health = processor.health_check()
        if not health["success"]:
            logger.error("Health check failed, but continuing with demo")
            logger.error(f"Health status: {health}")

        # Example document processing
        document_url = os.getenv(
            "DEMO_DOCUMENT_URL",
            "/home/mathewvkariath/Desktop/keycode_25/standin-ai/confluence.txt",
        )

        logger.info("Processing demo document...")
        result = processor.process_document(
            url=document_url, doc_id="demo-doc-1", store_vectors=True
        )

        if result["success"]:
            logger.info("Document processed successfully")
            logger.info(f"Document ID: {result['doc_id']}")
            logger.info(f"Processing time: {result['processing_time']:.2f}s")
            logger.info(f"Chunks created: {result['chunks_count']}")
            logger.info(f"Summary length: {result['summary_length']} characters")
            logger.info(f"Stored in vector DB: {result['stored_in_vector_db']}")
            logger.debug(f"Summary preview: {result['summary'][:200]}...")
        else:
            logger.error(f"Document processing failed: {result['error']}")
            return 1

        logger.info("Retrieving collection statistics...")
        stats = processor.get_collection_stats()
        if stats["success"]:
            logger.info("Collection stats retrieved successfully")
            logger.info(f"Collection: {stats['collection_name']}")
            logger.info(f"Vector size: {stats['vector_size']}")
            logger.info(f"Points count: {stats['points_count']}")
            logger.info(f"Distance metric: {stats['distance_metric']}")
        else:
            logger.error(f"Failed to get collection stats: {stats['error']}")

        # Generate narration
        try:
            logger.info("Generating narration...")
            summary_for_narration = Summary(
                title="Demo Document", content=result["summary"], attendee="Demo User"
            )
            narration = NarrationGenerator().generate(summary_for_narration)
            logger.info("Narration generated successfully")
            logger.info(f"Generated Narration with {len(narration.slides)} slides")
        except Exception as e:
            logger.error(f"Failed to generate narration: {e}", exc_info=True)

        logger.info("Workflow completed successfully")
        logger.info("Next steps:")
        logger.info("- Implement FastAPI endpoints using these services")
        logger.info("- Create search endpoints for document retrieval")

        return 0

    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


from fastapi import APIRouter
from fastapi.responses import JSONResponse
from models.models import MeetingCreationRequest
meeting_router = APIRouter()


@meeting_router.get("/create-meeting-narrations", tags=["Health"])
async def meeting_router(req: MeetingCreationRequest):
    return run_pipeline(req)


# app/endpoints/create_meeting_narration.py

from fastapi import APIRouter
from pydantic import BaseModel

meeting_router = APIRouter()


class MeetingRequest(BaseModel):
    url: str
    attendee: str
    duration: int


@meeting_router.post("/create_meeting_narration")
def create_meeting_narration(req: MeetingRequest):
    run_pipeline()
    # replace with your actual function call
    return {
        "message": f"Narration created for {req.attendee} ({req.duration} mins) with doc {req.url}"
    }
