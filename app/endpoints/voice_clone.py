from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import requests
import logging
import os

# Setup logging
logger = logging.getLogger("voice_clone")
logger.setLevel(logging.INFO)

# Router setup
voice_router = APIRouter()

# Constants
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
API_BASE_URL = "https://api.cartesia.ai/voices"
API_VERSION = "2025-04-16"


# Pydantic models
class VoiceCloneResponse(BaseModel):
    success: bool
    voice_id: Optional[str] = None
    message: str


@voice_router.post("/upload-and-clone", response_model=VoiceCloneResponse)
async def upload_and_clone_voice(
    audio_file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    language: str = Form("en"),
    enhance_audio: bool = Form(True),
):
    """Upload audio and create voice clone"""

    if not CARTESIA_API_KEY:
        raise HTTPException(status_code=500, detail="Cartesia API key not configured")

    try:
        files = {
            "clip": (
                audio_file.filename,
                await audio_file.read(),
                audio_file.content_type,
            )
        }
        payload = {
            "name": name,
            "description": description or "",
            "language": language,
            "enhance": str(enhance_audio).lower(),
        }
        headers = {
            "Cartesia-Version": API_VERSION,
            "Authorization": f"Bearer {CARTESIA_API_KEY}",
        }

        response = requests.post(
            f"{API_BASE_URL}/clone", data=payload, files=files, headers=headers
        )
        response.raise_for_status()

        return VoiceCloneResponse(
            success=True,
            voice_id=response.json().get("id"),
            message="Voice cloned successfully",
        )

    except Exception as e:
        logger.error(f"Error in voice cloning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")


@voice_router.get("/voices")
async def list_voices():
    """List all cloned voices"""
    try:
        headers = {
            "Cartesia-Version": API_VERSION,
            "Authorization": f"Bearer {CARTESIA_API_KEY}",
        }

        response = requests.get(API_BASE_URL, headers=headers)
        response.raise_for_status()

        return response.json()

    except Exception as e:
        logger.error(f"Error listing voices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list voices: {str(e)}")


@voice_router.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a cloned voice"""
    try:
        headers = {
            "Cartesia-Version": API_VERSION,
            "Authorization": f"Bearer {CARTESIA_API_KEY}",
        }

        response = requests.delete(f"{API_BASE_URL}/{voice_id}", headers=headers)
        response.raise_for_status()

        return {"success": True, "message": f"Voice {voice_id} deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting voice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete voice: {str(e)}")
