import os
import subprocess
from pathlib import Path
import signal

from fastapi import FastAPI

PROJECT_ROOT = "/Users/ratheesha/Desktop/Ratheesha/Keycode-2025-bitsandbytes/standin-ai"
PYTHON_BIN = f"{PROJECT_ROOT}/env/bin/python"
AGENT_SCRIPT = f"{PROJECT_ROOT}/src/agent_inititial.py"

app = FastAPI(title="StandIn AI Agent API")


@app.post("/start")
def start_agent():
    python = PYTHON_BIN if Path(PYTHON_BIN).exists() else "python3"
    proc = subprocess.Popen([python, AGENT_SCRIPT, "console"], cwd=PROJECT_ROOT, env=os.environ.copy())
    return {"started": True, "pid": proc.pid}

@app.post("/stop")
def stop_agent(pid: int):
    python = PYTHON_BIN if Path(PYTHON_BIN).exists() else "python3"
    os.kill(pid, signal.SIGTERM)
    return {"stopped": True}




