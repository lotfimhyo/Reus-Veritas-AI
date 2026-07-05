import asyncio
import uuid
from typing import Dict, Any, Optional
from agents.base import BaseAgent
from core.model_router import ModelRouter
from memory.store import Store

class AgentManager:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.router = ModelRouter()
        self.store = Store()

    async def create_agent(self, agent_type: str, name: Optional[str]) -> str:
        agent_id = str(uuid.uuid4())
        agent_name = name or f"{agent_type}-{agent_id[:6]}"
        # For now we only have software_engineer sample
        if agent_type == 'software_engineer' or agent_type == 'software-engineer':
            from agents.software_engineer.agent import SoftwareEngineerAgent
            agent = SoftwareEngineerAgent(id=agent_id, name=agent_name, manager=self)
        else:
            # fallback generic agent
            from agents.base import BaseAgent
            agent = BaseAgent(id=agent_id, name=agent_name, manager=self)
        self.agents[agent_id] = agent
        await self.store.save_agent_metadata(agent_id, {"type": agent_type, "name": agent_name})
        return agent_id

    async def list_agents(self):
        out = []
        for aid, agent in self.agents.items():
            out.append({"id": aid, "name": agent.name, "status": agent.status})
        return out

    async def start_agent(self, agent_id: str) -> bool:
        agent = self.agents.get(agent_id)
        if not agent:
            return False
        await agent.start()
        return True

    async def stop_agent(self, agent_id: str) -> bool:
        agent = self.agents.get(agent_id)
        if not agent:
            return False
        await agent.stop()
        return True

    async def dispatch_task(self, payload: Dict[str, Any]) -> str:
        # choose agent
        agent_id = payload.get('agent_id')
        if agent_id:
            agent = self.agents.get(agent_id)
        else:
            # choose by type
            agent_type = payload.get('agent_type')
            agent = None
            for a in self.agents.values():
                if a.agent_type == agent_type:
                    agent = a
                    break
        if not agent:
            # no agent -> create a new one on demand
            if payload.get('agent_type'):
                agent_id_new = await self.create_agent(payload.get('agent_type'), None)
                agent = self.agents.get(agent_id_new)
            else:
                raise ValueError('No agent available')
        task_id = await agent.enqueue_task(payload)
        return task_id


# singleton
manager = AgentManager()
