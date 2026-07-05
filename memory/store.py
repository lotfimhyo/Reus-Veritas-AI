import json
import asyncio
from pathlib import Path
from typing import Dict, Any

DATA_DIR = Path('data')
DATA_DIR.mkdir(parents=True, exist_ok=True)
AGENTS_FILE = DATA_DIR / 'agents.json'

class Store:
    def __init__(self):
        self._lock = asyncio.Lock()
        if not AGENTS_FILE.exists():
            AGENTS_FILE.write_text(json.dumps({}))

    async def save_agent_metadata(self, agent_id: str, metadata: Dict[str, Any]):
        async with self._lock:
            data = json.loads(AGENTS_FILE.read_text())
            data[agent_id] = metadata
            AGENTS_FILE.write_text(json.dumps(data, indent=2))

    async def load_all_agents(self):
        async with self._lock:
            data = json.loads(AGENTS_FILE.read_text())
            return data
