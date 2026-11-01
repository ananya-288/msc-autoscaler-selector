#!/bin/bash
# run_mech.sh - one experiment run in the FINAL environment (Linkerd + KEDA + VPA).
#
#   ./run_mech.sh tune-vpa                    # once per cluster: make the VPA recommender react in minutes
#   ./run_mech.sh hpa      burst 1            # HPA-only   (CPU 50 %, 1-5 replicas)
#   python calibrate_keda.py results/mech/hpa_burst_run1_metrics.csv   # after the first HPA run
#   ./run_mech.sh keda     burst 1            # KEDA-only  (request rate, target from calibration, 1-5 replicas)
#   ./run_mech.sh vertical burst 1            # vertical-only (VPA recommendation applied in place, 1 replica)
#   RESET_WAIT=30 ./run_mech.sh hpa burst test 3     # short rehearsal (3 minutes of the shape)
#   ./run_mech.sh selector eval_heldout 1     # live ML selector (FINAL: tuned ensemble, scale up vs out, 2-min hold)
#   SELECTOR=knn ./run_mech.sh selector ...   # earlier kNN selector (3 classes, gate 0.8)
#   ./run_mech.sh rules    eval_heldout 1     # live rule-based switcher (baseline)
#
# Run in the background:  nohup ./run_mech.sh hpa burst 1 > results/logs/mech_hpa_burst_1.out 2>&1 &
# redis-cart is never scaled. Every run starts from 1 replica and the default CPU requests.
set -u
MODE=${1:-}; SHAPE=${2:-}; RUN=${3:-}; MIN=${4:-15}
PY=${PY:-$HOME/venv/bin/python}
RESET_WAIT=${RESET_WAIT:-90}
TARGET=50; MAXR=5
cd "$(dirname "$0")"
SERVICES="adservice cartservice checkoutservice currencyservice emailservice frontend paymentservice productcatalogservice recommendationservice shippingservice"
PROM=http://prometheus.linkerd-viz.svc.cluster.local:9090

if [[ "$MODE" == "tune-vpa" ]]; then
  # native VPA recommender, tuned to react within minutes (defaults: 1 min loop, 24 h half-life)
  kubectl -n kube-system patch deployment vpa-recommender --type=json -p='[{"op":"add","path":"/spec/template/spec/containers/0/args","value":[
    "--recommender-interval=15s","--cpu-histogram-decay-half-life=2m","--target-cpu-percentile=0.9",
    "--pod-recommendation-min-cpu-millicores=50"]}]'
  kubectl -n kube-system rollout status deploy/vpa-recommender
  kubectl -n kube-system get deploy vpa-recommender -o jsonpath='{.spec.template.spec.containers[0].args}{"\n"}'
  exit 0
fi

if [[ ! "$MODE" =~ ^(hpa|keda|vertical|selector|rules)$ ]] || [[ -z "$SHAPE" || -z "$RUN" ]]; then
  sed -n '2,12p' "$0"; exit 1
fi
[[ -f "workloads/${SHAPE}.csv" ]] || { echo "STOP: workloads/${SHAPE}.csv not found"; exit 1; }
[[ "$MODE" =~ ^(keda|selector|rules)$ && ! -f keda_thresholds.json ]] && { echo "STOP: run calibrate_keda.py first"; exit 1; }

TAG="${MODE}_${SHAPE}_run${RUN}"
OUT=results/mech; mkdir -p $OUT results/locust results/logs
[[ -e "$OUT/${TAG}_metrics.csv" ]] && { echo "STOP: ${TAG} already exists, use another run number"; exit 1; }
FE=$(kubectl get svc frontend-external -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
[[ -z "$FE" ]] && { echo "STOP: cannot read frontend address"; exit 1; }

reset() {
  kubectl delete scaledobject --all --ignore-not-found >/dev/null
  kubectl delete hpa --all --ignore-not-found >/dev/null
  kubectl delete vpa --all --ignore-not-found >/dev/null
  pkill -f vertical_actuator.py 2>/dev/null; pkill -f live_selector.py 2>/dev/null
  kubectl scale deploy --all --replicas=1 >/dev/null
  kubectl rollout restart deploy >/dev/null          # new pods = default CPU requests again
  kubectl rollout status deploy --timeout=10m >/dev/null
}

echo "=== ${TAG}: workload ${SHAPE}, $(date -u '+%F %T') UTC  reset"
reset
echo "=== waiting ${RESET_WAIT}s"; sleep "$RESET_WAIT"

case "$MODE" in
hpa)
  for s in $SERVICES; do kubectl autoscale deploy "$s" --cpu="${TARGET}%" --min=1 --max=$MAXR >/dev/null; done
  ;;
keda)
  for s in $SERVICES; do
    THR=$("$PY" -c "import json;print(json.load(open('keda_thresholds.json')).get('$s',10))")
    cat <<EOF | kubectl apply -f - >/dev/null
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata: {name: ${s}-rps}
spec:
  scaleTargetRef: {name: ${s}}
  minReplicaCount: 1
  maxReplicaCount: ${MAXR}
  pollingInterval: 15
  triggers:
  - type: prometheus
    metadata:
      serverAddress: ${PROM}
      query: sum(rate(request_total{namespace="default",deployment="${s}",direction="inbound"}[1m]))
      threshold: "${THR}"
EOF
  done
  ;;
vertical)
  for s in $SERVICES; do
    cat <<EOF | kubectl apply -f - >/dev/null
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata: {name: ${s}-vpa}
spec:
  targetRef: {apiVersion: apps/v1, kind: Deployment, name: ${s}}
  updatePolicy: {updateMode: "Off"}
  resourcePolicy:
    containerPolicies:
    - {containerName: server, controlledResources: [cpu], minAllowed: {cpu: 50m}, maxAllowed: {cpu: 1000m}}
    - {containerName: linkerd-proxy, mode: "Off"}
EOF
  done
  "$PY" vertical_actuator.py --minutes $((MIN + 1)) --log "$OUT/${TAG}_vertical_actions.csv" > "results/logs/actuator_${TAG}.log" 2>&1 &
  ;;
selector|rules)
  SEL=$([[ "$MODE" == "selector" ]] && echo "${SELECTOR:-ensemble}" || echo rules)
  [[ "$SEL" == "ensemble" && ! -f "${FINAL:-results/selector/final_selector.json}" ]] && { echo "STOP: final_selector.json not found"; exit 1; }
  [[ "$SEL" == "ensemble" && ! -f results/selector/training_dataset_floor50_avg12_binary_lags.csv ]] && { echo "STOP: binary+lags training dataset not found"; exit 1; }
  [[ "$SEL" == "knn" && ! -f "${TRAIN:-results/selector/training_dataset_floor50_avg12.csv}" ]] && { echo "STOP: training dataset not found"; exit 1; }
  "$PY" vertical_actuator.py --minutes $((MIN + 1)) --log "$OUT/${TAG}_vertical_actions.csv" > "results/logs/actuator_${TAG}.log" 2>&1 &
  "$PY" live_selector.py --mode "$SEL" --minutes $((MIN + 1)) --log "$OUT/${TAG}_decisions.csv" \
      --train "${TRAIN:-results/selector/training_dataset_floor50_avg12.csv}" --gate "${GATE:-0.8}" \
      --final "${FINAL:-results/selector/final_selector.json}" \
      > "results/logs/selector_${TAG}.log" 2>&1 &
  sleep 20                                            # let it put every service on vertical first
  ;;
esac

START=$(date -u '+%F %T')
"$PY" collector.py --out "$OUT/${TAG}_metrics.csv" --minutes $((MIN + 1)) --mechanism "$MODE" > "results/logs/collector_${TAG}.log" 2>&1 &
COL=$!
WORKLOAD=workloads/${SHAPE}.csv timeout $((MIN * 60 + 30)) "$PY" -m locust -f locust_authors.py,trace_shape.py \
  --host "http://${FE}" --headless --csv="results/locust/${TAG}" --csv-full-history > "results/logs/locust_${TAG}.log" 2>&1 &
LOC=$!
printf 'run=%s\nmode=%s\nworkload=%s\nstart_utc=%s\nminutes=%s\ntarget_cpu=%s\nmax_replicas=%s\n' \
  "$TAG" "$MODE" "$SHAPE" "$START" "$MIN" "$TARGET" "$MAXR" > "$OUT/${TAG}_info.txt"
[[ "$MODE" == "keda" ]] && cp keda_thresholds.json "$OUT/${TAG}_keda_thresholds.json"
echo "=== ${START} UTC  started Locust (${SHAPE}) + collector; ends in about $((MIN + 1)) min"

wait "$LOC"; echo "=== $(date -u '+%T') UTC  Locust finished"
wait "$COL"; echo "=== $(date -u '+%T') UTC  collector finished"
pkill -f vertical_actuator.py 2>/dev/null; pkill -f live_selector.py 2>/dev/null

kubectl get events --sort-by=.lastTimestamp -o custom-columns=TIME:.lastTimestamp,REASON:.reason,OBJECT:.involvedObject.name,MSG:.message > "$OUT/${TAG}_events.txt"
kubectl get hpa,scaledobject,vpa -o wide > "$OUT/${TAG}_autoscalers.txt" 2>&1
kubectl get pods -o wide > "$OUT/${TAG}_pods.txt"
kubectl delete scaledobject --all --ignore-not-found >/dev/null; kubectl delete hpa --all --ignore-not-found >/dev/null
kubectl delete vpa --all --ignore-not-found >/dev/null

echo "=== DONE ${TAG}"
echo "    metrics rows: $(($(wc -l < "$OUT/${TAG}_metrics.csv") - 1))   collector errors: $(grep -c error "results/logs/collector_${TAG}.log")"
[[ "$MODE" =~ ^(vertical|selector|rules)$ ]] && echo "    resize actions: $(($(wc -l < "$OUT/${TAG}_vertical_actions.csv") - 1))"
[[ "$MODE" =~ ^(selector|rules)$ ]] && echo "    $(tail -n 1 "results/logs/selector_${TAG}.log")"
grep "Aggregated" "results/logs/locust_${TAG}.log" | tail -n 2
