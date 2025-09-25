# from fastapi import APIRouter
# from fastapi.responses import JSONResponse

# health_router = APIRouter()


# @health_router.get("/health", tags=["Health"])
# async def health_check():
#     return JSONResponse(content={"status": "ok"})
# app/endpoints/health.py
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}
