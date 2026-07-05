# Isolation instructions

By default docker-compose will run the service using NETWORK_MODE=bridge (connected to Docker bridge network).

To run fully isolated (no network access), set:

  NETWORK_MODE=none

in your .env file before running docker-compose up.

Example:

  cp .env.example .env
  # Edit .env and set NETWORK_MODE=none if isolation is required
  docker-compose up --build

Note: The default in this repository is non-isolated (bridge) as requested.
