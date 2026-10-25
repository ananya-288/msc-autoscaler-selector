#!/bin/bash
# run_experiment.sh (v2) - one complete experiment run on the EC2 runner.
#
#   ./run_experiment.sh MODE RUN [MINUTES] [USERS]
#   MODE = hpa | smarthpa | prosmart
#
# Scenario is set with environment variables (defaults = paper's 5R-50%):
#   MAXR=2 TARGET=20 ./run_experiment.sh smarthpa 2 15 300     # scenario 2R-20%
#   MAXR=2 TARGET=20 ./run_experiment.sh hpa      2 15 300
#   ./run_experiment.sh hpa test 2 20                          # 2-min rehearsal
#   MAXR=2 TARGET=20 PS_MODELS=models_2R20 ./run_experiment.sh prosmart 2 15 300
#
# Run in the background so a dropped SSH session does not stop it:
#   nohup ./run_experiment.sh smarthpa 2 15 300 > results/logs/run.out 2>&1 &
#   tail -f results/logs/run.out

set -u
MODE=${1:-}; RUN=${2:-}; MIN=${3:-15}; USERS=${4:-600}
MAXR=${MAXR:-5}; TARGET=${TARGET:-50}
PY=${PY:-$HOME/venv/bin/python}             # the venv Python
SMART_REPO=${SMART_REPO:-$HOME/Smart_HPA}   # cloned Smart_HPA repo
PROSMART_REPO=${PROSMART_REPO:-$HOME/ProSmart_HPA}
PS_MODELS=${PS_MODELS:-models_2R20}         # retrained PROPHET models (train_prosmart_models.py)
WINDOW=${WINDOW:-25}                        # ProSmart prediction window (s)
RESET_WAIT=${RESET_WAIT:-180}
SCEN="${MAXR} Replicas + ${TARGET}%"
cd "$(dirname "$0")"

if [[ "$MODE" != "hpa" && "$MODE" != "smarthpa" && "$MODE" != "prosmart" ]] || [[ -z "$RUN" ]]; then
  echo "Usage: [MAXR=n TARGET=n] ./run_experiment.sh hpa|smarthpa|prosmart RUN [MINUTES] [USERS]"; exit 1
fi

TAG="${MODE}_${MAXR}R-${TARGET}_run${RUN}"
KB_OUT="results/${MODE}/Test/${SCEN}/Run ${RUN}"
mkdir -p results/locust results/cluster results/logs "results/${MODE}/Test/${SCEN}"
if [[ -e "$KB_OUT" || -e "results/locust/${TAG}_stats.csv" ]]; then
  echo "STOP: results for ${TAG} already exist. Use another run number."; exit 1
fi

FE=$(kubectl get svc frontend-external -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
[[ -z "$FE" ]] && { echo "STOP: cannot read frontend address (is kubectl working?)"; exit 1; }
HOST="http://${FE}"

echo "=== ${TAG}: scenario ${SCEN}, ${MIN} min, ${USERS} users, host ${HOST}"
echo "=== $(date -u '+%F %T') UTC  reset: delete HPAs, scale everything to 1"
kubectl delete hpa --all
kubectl scale deployment --all --replicas=1
echo "=== waiting ${RESET_WAIT}s for the app to settle"
sleep "$RESET_WAIT"

if [[ "$MODE" == "hpa" ]]; then
  for d in $(kubectl get deploy -o name); do
    kubectl autoscale "$d" --cpu="${TARGET}%" --min=1 --max="$MAXR"
  done
  "$PY" hpa_logger.py --minutes "$MIN" --out "$KB_OUT" > "results/logs/controller_${TAG}.log" 2>&1 &
elif [[ "$MODE" == "smarthpa" ]]; then
  "$PY" prepare_smart_hpa.py "$SMART_REPO" --minutes "$MIN" --force > "results/logs/prepare_${TAG}.log" 2>&1 \
    || { echo "STOP: prepare failed, see results/logs/prepare_${TAG}.log"; exit 1; }
  # apply the scenario to all 11 Microservice Managers
  N=$(grep -lE "target_cpu = 50|max_replica = 5" smart_hpa_run/*.py | wc -l)
  sed -i -E "s/target_cpu = 50/target_cpu = ${TARGET}/; s/max_replica = 5/max_replica = ${MAXR}/" smart_hpa_run/*.py
  echo "=== scenario patched into ${N} manager files:"
  grep -hE "^\s*(target_cpu|max_replica) =" smart_hpa_run/frontend.py
  [[ "$N" -ne 11 ]] && echo "WARNING: expected 11 manager files, found ${N}"
  ( cd smart_hpa_run && "$PY" run_smart_hpa.py ) > "results/logs/controller_${TAG}.log" 2>&1 &
else
  "$PY" prepare_prosmart.py "$PROSMART_REPO" --models "$PS_MODELS" --minutes "$MIN" \
      --maxr "$MAXR" --target "$TARGET" --window "$WINDOW" --force > "results/logs/prepare_${TAG}.log" 2>&1 \
    || { echo "STOP: prepare_prosmart.py failed, see results/logs/prepare_${TAG}.log"; exit 1; }
  cat "results/logs/prepare_${TAG}.log"
  ( cd prosmart_run && "$PY" run_prosmart_hpa.py ) > "results/logs/controller_${TAG}.log" 2>&1 &
fi
CTRL=$!

START=$(date -u '+%F %T')
"$PY" -m locust -f locust_authors.py --host "$HOST" --headless -u "$USERS" -r 2 -t "${MIN}m" \
  --csv="results/locust/${TAG}" --csv-full-history > "results/logs/locust_${TAG}.log" 2>&1 &
LOC=$!
echo "=== ${START} UTC  started controller (pid ${CTRL}) and Locust (pid ${LOC})"
printf 'run=%s\nscenario=%s\nstart_utc=%s\nminutes=%s\nusers=%s\nhost=%s\n' \
  "$TAG" "$SCEN" "$START" "$MIN" "$USERS" "$HOST" > "results/logs/info_${TAG}.txt"

# record replica counts every 15 s (full scaling history, not just the end state)
( while kill -0 "$LOC" 2>/dev/null; do
    echo "# $(date -u '+%T')"; kubectl get deploy --no-headers -o custom-columns=N:.metadata.name,R:.status.replicas
    sleep 15; done ) > "results/cluster/${TAG}_replicas_timeline.txt" 2>&1 &

wait "$LOC";  echo "=== $(date -u '+%T') UTC  Locust finished (exit $?)"
wait "$CTRL"; CTRL_EXIT=$?
echo "=== $(date -u '+%T') UTC  controller finished (exit ${CTRL_EXIT})"
[[ $CTRL_EXIT -ne 0 ]] && echo "WARNING: controller error, see results/logs/controller_${TAG}.log"

echo "=== saving cluster state"
kubectl get events --sort-by=.lastTimestamp \
  -o custom-columns=TIME:.lastTimestamp,TYPE:.type,REASON:.reason,OBJECT:.involvedObject.name,MSG:.message \
  > "results/cluster/${TAG}_events.txt"
kubectl get pods -o wide > "results/cluster/${TAG}_pods.txt"
if [[ "$MODE" == "hpa" ]]; then
  kubectl get hpa > "results/cluster/${TAG}_hpa.txt"; kubectl delete hpa --all
elif [[ "$MODE" == "smarthpa" ]]; then
  cp -r "smart_hpa_run/Knowledge Base" "$KB_OUT"
  echo "=== non-empty ARM decisions: $(grep -c 'ARM_decision \[[^]]' "results/logs/controller_${TAG}.log")"
else
  cp -r "prosmart_run/Knowledge Base" "$KB_OUT"
  echo "=== scale commands: $(grep -c ' scaled' "results/logs/controller_${TAG}.log")   errors: $(grep -c -E 'Traceback|Error' "results/logs/controller_${TAG}.log")"
fi

echo "=== DONE ${TAG}. Summary:"
tail -n 25 "results/logs/locust_${TAG}.log" | grep -E "Aggregated|Error|occurrences" || true
