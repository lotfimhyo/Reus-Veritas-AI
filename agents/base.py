import asyncio
import uuid
from typing import Any, Dict

class BaseAgent:
    def __init__(self, id: str, name: str, manager=None):
        self.id = id
        self.name = name
        self.manager = manager
        self.status = 'created'
        self.agent_type = 'generic'
        self.task_queue = asyncio.Queue()
        self._runner = None

    async def start(self):
        if self.status == 'running':
            return
        self.status = 'running'
        self._runner = asyncio.create_task(self._worker())

    async def stop(self):
        self.status = 'stopping'
        if self._runner:
            self._runner.cancel()
            try:
                await self._runner
            except asyncio.CancelledError:
                pass
        self.status = 'stopped'

    async def enqueue_task(self, task: Dict[str, Any]) -> str:
        task_id = str(uuid.uuid4())
        await self.task_queue.put({"id": task_id, "payload": task})
        return task_id

    async def _worker(self):
        while True:
            item = await self.task_queue.get()
            try:
                await self.handle_task(item)
            except Exception as e:
                print(f"Agent {self.name} error: {e}")

    async def handle_task(self, item: Dict[str, Any]):
        # Default behaviour: echo the task
        print(f"Agent {self.name} handling task {item['id']}")
