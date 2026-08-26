#!/bin/bash
set -u
ENVF=$1; CON=$2; TAG=$3; shift 3
exec 9>"/tmp/gpu_${TAG}.lock"; flock -n 9 || exit 1
cd /home/minseok/ai-forensics
for spec in "$@"; do
  OBJ=${spec%%:*}; TRIG=${spec##*:}
  echo "$(date +%H:%M:%S) START $OBJ/$TRIG [$TAG]" >> /tmp/grid_progress.log
  docker compose --env-file "$ENVF" -f docker/docker-compose.yml exec -T -e HF_HUB_OFFLINE=1 aiforensics \
    python scripts/run_objective_attack.py --objective "$OBJ" --trigger "$TRIG" --poison-rate 0.5 \
    > "/tmp/obj_${OBJ}_${TRIG}.log" 2>&1
  echo "$(date +%H:%M:%S) DONE  $OBJ/$TRIG rc=$? [$TAG]" >> /tmp/grid_progress.log
done
echo "$(date +%H:%M:%S) $TAG FINISHED" >> /tmp/grid_progress.log
