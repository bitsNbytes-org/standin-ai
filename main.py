#create fast api app with health check endpoint and create meeting narration endpoint
from fastapi import FastAPI
from app.endpoints.health import router as health_router
from app.endpoints.create_meeting_narration import meeting_router
from app.endpoints.voice_clone import voice_router
app = FastAPI() 
app.include_router(health_router)
app.include_router(meeting_router)
app.include_router(voice_router)