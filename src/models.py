"""Data models for video narration generation."""

from typing import List
from pydantic import BaseModel, Field
from datetime import datetime


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
    
