#!/usr/bin/env bash
set -euo pipefail

docker run -d \
  --name postgresql \
  --shm-size=1g \
  -e POSTGRES_USER=<DB_USER> \
  -e POSTGRES_PASSWORD=<DB_PASSWORD> \
  -e POSTGRES_DB=rags_authorization \
  -p 5433:5432 \
  -v postgresql-data:/var/lib/postgresql/data \
  pgvector/pgvector:pg16

echo "Waiting for Postgres to accept connections..."
until docker exec postgresql pg_isready -U <DB_USER> -d rags_authorization >/dev/null 2>&1; do
  sleep 1
done

docker exec postgresql psql -U <DB_USER> -d rags_authorization -c "CREATE EXTENSION IF NOT EXISTS vector;"
