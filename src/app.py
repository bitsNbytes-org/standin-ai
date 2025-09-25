import os
import json
import subprocess
from pathlib import Path
import signal
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel, Field

PROJECT_ROOT = "/Users/ratheesha/Desktop/Ratheesha/Keycode-2025-bitsandbytes/standin-ai"
PYTHON_BIN = f"{PROJECT_ROOT}/env/bin/python"
AGENT_SCRIPT = f"{PROJECT_ROOT}/src/agent_inititial.py"

app = FastAPI(title="StandIn AI Agent API")

class StartPayload(BaseModel):
    status: str = "completed"
    narration: Dict[str, Any] = Field(default_factory=dict)

@app.post("/start")
def start_agent(payload: StartPayload):
    python = PYTHON_BIN if Path(PYTHON_BIN).exists() else "python3"
    # Serialize payload once so quoting is safe when passed as a single arg
    payload_arg = json.dumps(payload.dict())
    print(payload_arg)
    proc = subprocess.Popen(
        [python, AGENT_SCRIPT, "console", "--payload", payload_arg],
        cwd=PROJECT_ROOT,
        env=os.environ.copy()
    )
    return {"started": True, "pid": proc.pid}

@app.post("/stop")
def stop_agent(pid: int):
    os.kill(pid, signal.SIGTERM)
    return {"stopped": True}
