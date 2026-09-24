#!/usr/bin/env bash
# Start the Milvus standalone server the bake-off measures (bakeoff/contenders/milvus.py).
#
# Both -v mounts below are load-bearing: the image ships neither embedEtcd.yaml nor user.yaml,
# and with ETCD_USE_EMBED=true and no embedEtcd.yaml the embedded etcd segfaults on startup --
# the container dies in under a second with exit 134. Recreating it by hand without them is
# the usual way this breaks.
#
#   ./docker/milvus/start.sh           reuse the existing data volume
#   ./docker/milvus/start.sh --fresh   discard it and start from scratch
set -euo pipefail

CONTAINER=milvus-bakeoff
VOLUME=milvus-bakeoff-v3
IMAGE=milvusdb/milvus:v3.0.1
CONFIGS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
[ "${1:-}" = "--fresh" ] && docker volume rm "$VOLUME" >/dev/null 2>&1 || true

docker run -d \
    --name "$CONTAINER" \
    --restart unless-stopped \
    --security-opt seccomp:unconfined \
    -e ETCD_USE_EMBED=true \
    -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
    -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
    -e COMMON_STORAGETYPE=local \
    -e DEPLOY_MODE=STANDALONE \
    -v "$VOLUME":/var/lib/milvus \
    -v "$CONFIGS/embedEtcd.yaml":/milvus/configs/embedEtcd.yaml \
    -v "$CONFIGS/user.yaml":/milvus/configs/user.yaml \
    -p 19530:19530 -p 9091:9091 \
    --health-cmd="curl -f http://localhost:9091/healthz" \
    --health-interval=10s --health-start-period=90s \
    --health-timeout=20s --health-retries=3 \
    "$IMAGE" milvus run standalone >/dev/null

printf 'waiting for %s' "$CONTAINER"
for _ in $(seq 1 60); do
    case "$(docker inspect "$CONTAINER" --format '{{.State.Health.Status}}')" in
        healthy) echo " -- healthy on localhost:19530"; exit 0 ;;
    esac
    if [ "$(docker inspect "$CONTAINER" --format '{{.State.Status}}')" != running ]; then
        echo " -- DIED (exit $(docker inspect "$CONTAINER" --format '{{.State.ExitCode}}'))"
        docker logs --tail 30 "$CONTAINER"
        exit 1
    fi
    printf '.'; sleep 5
done
echo " -- never became healthy"; exit 1
