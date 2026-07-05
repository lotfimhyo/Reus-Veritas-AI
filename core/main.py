from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
from typing import Dict, Optional
import asyncio
from core.agent_manager import AgentManager

app = FastAPI(title="Reus-Veritas OS API")
manager = AgentManager()

class AgentCreate(BaseModel):
    type: str
    name: Optional[str] = None

@app.post('/agents')
async def create_agent(req: AgentCreate):
    agent_id = await manager.create_agent(req.type, req.name)
    return {"id": agent_id}

@app.get('/agents')
async def list_agents():
    return await manager.list_agents()

@app.post('/agents/{agent_id}/start')
async def start_agent(agent_id: str):
    ok = await manager.start_agent(agent_id)
    if not ok:
        raise HTTPException(status_code=404, detail='Agent not found or could not be started')
    return {"status": "started"}

@app.post('/agents/{agent_id}/stop')
async def stop_agent(agent_id: str):
    ok = await manager.stop_agent(agent_id)
    if not ok:
        raise HTTPException(status_code=404, detail='Agent not found or could not be stopped')
    return {"status": "stopped"}

@app.post('/tasks')
async def submit_task(payload: Dict):
    # payload should include target agent_id or type
    target = payload.get('agent_id') or payload.get('agent_type')
    if not target:
        raise HTTPException(status_code=400, detail='agent_id or agent_type required')
    task_id = await manager.dispatch_task(payload)
    return {"task_id": task_id}

@app.get('/')
async def root():
    return {"service": "Reus-Veritas OS", "status": "ok"}
