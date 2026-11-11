#!/bin/bash
# run_batch.sh - run many evaluation runs back to back, unattended.
#
#   END_BY=17:40 nohup ./run_batch.sh eval_heldout 1 5 > results/logs/batch_$(date +%H%M).out 2>&1 &
#       -> runs 1..5 of all five methods on eval_heldout, interleaved, until 17:40 UTC
#
# For every round r (run number) the five methods run once, in a rotated order
# (round 1: vertical hpa keda rules selector; round 2: hpa keda rules selector vertical; ...),
# so time-of-day effects are spread evenly over the methods.
# Before each run : skips runs that already exist; checks every shop pod has its Linkerd proxy (2/2)
#                   and restarts the deployments if not; stops if the next run cannot finish by END_BY.
# After each run  : checks the run finished (DONE, collector errors 0), writes one summary line to
#                   results/logs/batch_summary.txt and makes a backup ~/backups/shpa_<time>.tar.gz
#                   (keeps the 3 newest).
# At the end      : if SCALE_DOWN=1 (default), scales the node group to 0.
# END_BY is a UTC clock time (HH:MM). Set it about 15 minutes before the lab session ends.
# Optional:
#   NTFY_TOPIC=<secret-topic>  push a phone notification after every run, on any problem and at the end
#                              (free ntfy app; subscribe to the same topic name; use a long random name)
#   S3_BUCKET=<bucket-name>    also copy each backup to S3, so results survive even if the lab session ends
set -u
SHAPE=${1:-}; FIRST=${2:-}; LAST=${3:-}
METHODS=(vertical hpa keda rules selector)
RUN_MIN=${RUN_MIN:-21}                 # minutes one run needs, including reset and saving
END_BY=${END_BY:-}
SCALE_DOWN=${SCALE_DOWN:-1}
cd "$(dirname "$0")"
[[ -z "$SHAPE" || -z "$FIRST" || -z "$LAST" ]] && { sed -n '2,16p' "$0"; exit 1; }
[[ -f "workloads/${SHAPE}.csv" ]] || { echo "STOP: workloads/${SHAPE}.csv not found"; exit 1; }
[[ -z "$END_BY" ]] && { echo "STOP: set END_BY=HH:MM (UTC), about 15 min before the lab session ends"; exit 1; }
END_TS=$(date -u -d "today $END_BY" +%s) || exit 1
mkdir -p ~/backups results/logs
SUM=results/logs/batch_summary.txt
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$SUM"; }
notify() {   # $1 = title, $2 = message ; silent if NTFY_TOPIC is not set
  [[ -z "${NTFY_TOPIC:-}" ]] && return 0
  curl -s -m 10 -H "Title: $1" -d "$2" "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
}

pods_ok() {   # every Online Boutique pod Running with all containers ready and 2 containers (app + linkerd-proxy)
  kubectl get pods -n default --no-headers 2>/dev/null | awk '$3!="Completed"{split($2,a,"/"); if($3!="Running"||a[1]!=a[2]||a[2]<2) bad=1} END{exit bad}'
}

notify "batch start" "$SHAPE runs $FIRST..$LAST, until $END_BY UTC"
log "batch start: shape=$SHAPE runs $FIRST..$LAST, stop starting new runs after $(date -u -d @$((END_TS - RUN_MIN*60)) +%H:%M) UTC"
done_n=0
for r in $(seq "$FIRST" "$LAST"); do
  off=$(( (r - 1) % ${#METHODS[@]} ))
  order=("${METHODS[@]:$off}" "${METHODS[@]:0:$off}")
  for m in "${order[@]}"; do
    tag="${m}_${SHAPE}_run${r}"
    if [[ -e "results/mech/${tag}_metrics.csv" ]]; then log "skip $tag (already exists)"; continue; fi
    now=$(date -u +%s)
    if (( now + RUN_MIN*60 > END_TS )); then log "STOP: not enough time left for $tag"; notify "batch stopped" "not enough session time left for $tag"; break 2; fi
    if ! pods_ok; then
      log "pods not all 2/2 - restarting deployments"
      kubectl rollout restart deploy -n default >/dev/null; for i in $(seq 40); do sleep 15; pods_ok && break; done
      pods_ok || { log "STOP: pods still not healthy"; notify "PROBLEM pods" "pods not healthy - batch stopped"; break 2; }
    fi
    log "start $tag"
    ./run_mech.sh "$m" "$SHAPE" "$r" > "results/logs/mech_${m}_${SHAPE}_${r}.out" 2>&1
    out="results/logs/mech_${m}_${SHAPE}_${r}.out"
    if grep -q "=== DONE" "$out"; then
      rows=$(grep -o "metrics rows: [0-9]*" "$out" | awk '{print $3}')
      errs=$(grep -o "collector errors: [0-9]*" "$out" | awk '{print $3}')
      agg=$(grep "Aggregated" "$out" | head -n 1 | awk '{print $2, "requests,", $3, "failures"}')
      log "done  $tag  rows=$rows collector_errors=$errs  $agg"
      done_n=$((done_n + 1))
      if [[ "$rows" != "704" || "$errs" != "0" ]]; then notify "CHECK $tag" "rows=$rows collector_errors=$errs"
      else notify "done $tag" "rows=$rows, $agg"; fi
    else
      log "PROBLEM: $tag did not finish - see $out"; tail -n 5 "$out" >> "$SUM"
      notify "PROBLEM $tag" "$(tail -n 3 "$out")"
    fi
    bk=~/backups/shpa_$(date -u +%m%d_%H%M).tar.gz
    ( cd ~ && tar -czf "$bk" --exclude='__pycache__' shpa ) \
      && ls -1t ~/backups/shpa_*.tar.gz | tail -n +4 | xargs -r rm -f
    if [[ -n "${S3_BUCKET:-}" ]]; then
      aws s3 cp "$bk" "s3://${S3_BUCKET}/backups/" --only-show-errors || log "WARNING: S3 backup failed"
    fi
  done
done
log "batch end: $done_n runs completed; newest backup: $(ls -1t ~/backups/shpa_*.tar.gz 2>/dev/null | head -n 1)"
if [[ "$SCALE_DOWN" == "1" ]]; then
  aws eks update-nodegroup-config --cluster-name shpa --nodegroup-name ng \
      --scaling-config minSize=0,maxSize=5,desiredSize=0 >/dev/null 2>&1 \
    && { log "node group scaled to 0"; notify "batch end" "$done_n runs completed; nodes scaled to 0"; } \
    || { log "WARNING: scale-down failed - do it by hand"; notify "SCALE-DOWN FAILED" "scale the node group to 0 by hand now"; }
fi
