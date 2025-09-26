# StandIn AI - Document Processing and Voice Cloning Service

A Python-based service that provides document processing, summarization, and voice cloning capabilities using various AI services including OpenAI, Cartesia, and vector database integration.

## Features

- **Document Processing**
  - Support for PDF and text files
  - URL support (HTTP, S3, Google Cloud Storage)
  - Text extraction and chunking
  - Intelligent document type detection

- **Summary Generation**
  - AI-powered document summarization
  - Configurable chunking and processing
  - Vector database integration for storage and retrieval
  - OpenAI integration for advanced text processing

- **Voice Cloning**
  - Upload and clone voices using Cartesia AI
  - Voice management (list, delete)
  - Support for multiple audio formats (WAV, MP3, OGG)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/standin-ai.git
cd standin-ai
```

2. Create and activate a virtual environment:
```bash
python -m venv standinai
source standinai/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with the following variables:

```plaintext
OPENAI_API_KEY=your_openai_key
CARTESIA_API_KEY=your_cartesia_key
QDRANT_API_KEY=your_qdrant_key
QDRANT_URL=your_qdrant_url
QDRANT_COLLECTION=dev-collection
QDRANT_VECTOR_SIZE=3072
```

## Usage

### Summary Generation

```python
from app.service.summary_service import SummaryService, SummaryConfig

# Initialize the service
config = SummaryConfig()
service = SummaryService(config)

# Generate summary
result = service.generate_summary("Your text content here", doc_id="optional-doc-id")
print(result.summary)
```

### Voice Cloning

```python
from fastapi import FastAPI
from app.endpoints.voice_clone import voice_router

app = FastAPI()
app.include_router(voice_router)

# Available endpoints:
# POST /upload-and-clone - Clone a voice
# GET /voices - List all voices
# DELETE /voices/{voice_id} - Delete a voice
```

## API Endpoints

### Voice Cloning

- `POST /upload-and-clone`
  - Clone a voice from audio file
  - Parameters:
    - `audio_file`: Audio file (WAV, MP3, OGG)
    - `name`: Voice name
    - `description`: Optional description
    - `language`: Language code (default: "en")
    - `enhance_audio`: Boolean (default: true)

- `GET /voices`
  - List all cloned voices

- `DELETE /voices/{voice_id}`
  - Delete a specific voice clone

## Dependencies

- FastAPI
- OpenAI
- Pydantic
- PyPDF
- Requests
- Boto3 (optional, for S3)
- Google Cloud Storage (optional)
- Qdrant Client

## Development
Start the development server:
```bash
uvicorn main:app --reload
```