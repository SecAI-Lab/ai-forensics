#!/bin/bash
# ONE script per GPU, a fixed list, strictly sequential. No concurrent waiters:
# three of them each saw "GPU free" at the same instant and all launched, which
# put three processes on one card and OOMed the third.
set -u
ENVF=$1; CON=$2; TAG=$3; shift 3
LOCK="/tmp/gpu_${TAG}.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "$(date +%H:%M:%S) $TAG already locked, refusing to start" >> /tmp/grid_progress.log; exit 1; }
cd /home/minseok/ai-forensics
for obj in "$@"; do
  echo "$(date +%H:%M:%S) START objective:$obj [$TAG]" >> /tmp/grid_progress.log
  docker compose --env-file "$ENVF" -f docker/docker-compose.yml exec -T -e HF_HUB_OFFLINE=1 aiforensics \
    python scripts/run_objective_attack.py --objective "$obj" --trigger word --poison-rate 0.5 \
    > "/tmp/obj_${obj}.log" 2>&1
  echo "$(date +%H:%M:%S) DONE  objective:$obj rc=$? [$TAG]" >> /tmp/grid_progress.log
done
echo "$(date +%H:%M:%S) $TAG FINISHED" >> /tmp/grid_progress.log
