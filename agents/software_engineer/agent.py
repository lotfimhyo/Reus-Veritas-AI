import asyncio
from agents.base import BaseAgent

class SoftwareEngineerAgent(BaseAgent):
    def __init__(self, id: str, name: str, manager=None):
        super().__init__(id, name, manager)
        self.agent_type = 'software_engineer'

    async def handle_task(self, item):
        print(f"[SoftwareEngineer] {self.name} received task {item['id']}")
        # Simulate work
        await asyncio.sleep(1)
        print(f"[SoftwareEngineer] {self.name} completed task {item['id']}")
