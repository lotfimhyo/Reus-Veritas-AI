# Basic example to create an agent via HTTP

curl -X POST "http://localhost:8080/agents" -H "Content-Type: application/json" -d '{"type": "software_engineer", "name": "alice"}'

# Start agent
# curl -X POST http://localhost:8080/agents/{id}/start

