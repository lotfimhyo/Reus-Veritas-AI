# Reus‑Veritas OS

Autonomous AI Operating System — The first OS for autonomous intelligent agents.

Arabic / English quick start

هذا المشروع هو إعادة بناء و إعادة تسمية لمستودع Reus-Veritas-AI ليصبح "Reus‑Veritas OS" — نظام تشغيل وطبقة إدارة فوق نماذج الذكاء الاصطناعي، مخصّص لتشغيل، إدارة، ومراقبة وكلاء مستقلين.

Quick start (developer):

1. Create virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run locally:

```bash
uvicorn core.main:app --reload --port 8080
```

3. Example: create an agent

```bash
curl -X POST "http://localhost:8080/agents" -H "Content-Type: application/json" -d '{"type": "software_engineer", "name": "alice"}'
```

Goals
- Host many specialized agents (Software Engineer, Research, Financial, Cybersecurity, etc.)
- Provide lifecycle management (create/start/stop/update)
- Provide Model Router to select best model per task (stubbed)
- Provide Memory system for short/long term memory
- Provide hooks for Energy Optimizer, Evolution Engine, Security Core, Plugins, Automation

Repository layout (initial)

- core/  — kernel, agent manager, model router, API
- agents/ — example agent implementations
- memory/ — simple persistent store for agents and memories
- plugins/ — plugin engine skeleton
- docs/ — architecture & user guides
- preserved_old/ — archived legacy files from previous project

For detailed architecture and developer guide see ARCHITECTURE.md and docs/.

