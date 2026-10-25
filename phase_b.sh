#!/bin/bash
# phase_b.sh - install and verify the Phase B stack, one section at a time.
#   ./phase_b.sh linkerd      # 1. Linkerd control plane (service mesh)
#   ./phase_b.sh viz          # 2. Linkerd viz = Prometheus with per-service request metrics
#   ./phase_b.sh mesh         # 3. add the Linkerd proxy to all Online Boutique pods
#   ./phase_b.sh keda         # 4. KEDA
#   ./phase_b.sh vpa          # 5. VPA (we only use its recommender, updateMode Off)
#   ./phase_b.sh versions     # 6. record every version for the report
#   ./phase_b.sh test-rps     # check: per-service request rate is visible
#   ./phase_b.sh test-resize  # check: in-place CPU resize works without restart
#   ./phase_b.sh test-keda    # check: KEDA scales frontend on request rate
# Run each section, check its output, then run the next one.
set -euo pipefail
NS=default
PROM=http://prometheus.linkerd-viz.svc.cluster.local:9090
mkdir -p results/env

case "${1:-}" in
linkerd)
  if ! command -v linkerd >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSfL https://run.linkerd.io/install-edge | LINKERD2_VERSION=edge-26.10.1 sh
    kubectl apply --server-side -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.5.1/standard-install.yaml
    echo 'export PATH=$HOME/.linkerd2/bin:$PATH' >> ~/.bashrc
  fi
  export PATH=$HOME/.linkerd2/bin:$PATH
  linkerd version --client
  linkerd check --pre || { echo ">>> fix what 'check --pre' reports (see notes), then re-run"; exit 1; }
  linkerd install --crds | kubectl apply -f -
  linkerd install | kubectl apply -f -
  linkerd check
  ;;
viz)
  export PATH=$HOME/.linkerd2/bin:$PATH
  linkerd viz install | kubectl apply -f -
  linkerd viz check
  # allow in-cluster clients (KEDA, data collector) to query viz Prometheus; without this KEDA gets HTTP 403
  cat <<'EOF2' | kubectl apply -f -
apiVersion: policy.linkerd.io/v1alpha1
kind: NetworkAuthentication
metadata: {name: cluster-network, namespace: linkerd-viz}
spec:
  networks:
  - cidr: 172.31.0.0/16
---
apiVersion: policy.linkerd.io/v1alpha1
kind: AuthorizationPolicy
metadata: {name: prometheus-from-cluster, namespace: linkerd-viz}
spec:
  targetRef: {group: policy.linkerd.io, kind: Server, name: prometheus-admin}
  requiredAuthenticationRefs:
  - {group: policy.linkerd.io, kind: NetworkAuthentication, name: cluster-network}
EOF2
  ;;
mesh)
  # proxy CPU/memory requests are needed because HPA computes utilisation over ALL
  # containers of a pod; identical for every autoscaler, so comparisons stay fair
  kubectl annotate ns $NS linkerd.io/inject=enabled --overwrite
  kubectl annotate ns $NS config.linkerd.io/proxy-cpu-request=25m config.linkerd.io/proxy-memory-request=20Mi --overwrite
  kubectl rollout restart deploy -n $NS
  kubectl rollout status deploy -n $NS --timeout=10m
  kubectl get pods -n $NS          # every pod should now show 2/2 (app + linkerd-proxy)
  ;;
keda)
  helm repo add kedacore https://kedacore.github.io/charts >/dev/null; helm repo update >/dev/null
  helm upgrade --install keda kedacore/keda -n keda --create-namespace --version 2.21.0 --wait
  kubectl get pods -n keda
  ;;
vpa)
  [[ -d ~/autoscaler ]] || git clone --depth 1 https://github.com/kubernetes/autoscaler.git ~/autoscaler
  ( cd ~/autoscaler && git fetch --depth 1 origin tag vertical-pod-autoscaler-1.8.0 )
  ( cd ~/autoscaler/vertical-pod-autoscaler && ./hack/vpa-up.sh )
  kubectl get pods -n kube-system | grep vpa
  ;;
versions)
  export PATH=$HOME/.linkerd2/bin:$PATH
  { date -u; kubectl version; linkerd version; helm list -A;
    kubectl get deploy -n kube-system -o custom-columns=NAME:.metadata.name,IMAGE:.spec.template.spec.containers[0].image | grep -E "vpa|metrics";
    kubectl get nodes -o wide; } > results/env/versions_$(date +%F).txt 2>&1
  cat results/env/versions_$(date +%F).txt
  ;;
test-rps)
  echo "Per-service inbound request rate (needs some traffic, e.g. a short Locust run):"
  kubectl -n linkerd-viz port-forward svc/prometheus 9090:9090 >/dev/null 2>&1 & PF=$!; sleep 3
  curl -s localhost:9090/api/v1/query --data-urlencode \
    'query=sum by (deployment) (rate(request_total{namespace="default",direction="inbound"}[1m]))' \
    | python3 -c "import sys,json; [print(f\"{r['metric'].get('deployment','?'):24s} {float(r['value'][1]):8.2f} req/s\") for r in json.load(sys.stdin)['data']['result']]"
  kill $PF
  ;;
test-resize)
  P=$(kubectl get pod -n $NS -l app=currencyservice -o jsonpath='{.items[0].metadata.name}')
  echo "pod $P before:"; kubectl get pod $P -n $NS -o jsonpath='{.spec.containers[?(@.name=="server")].resources}{"\n"}restarts={.status.containerStatuses[?(@.name=="server")].restartCount}{"\n"}'
  kubectl patch pod $P -n $NS --subresource resize --patch \
    '{"spec":{"containers":[{"name":"server","resources":{"requests":{"cpu":"200m"},"limits":{"cpu":"400m"}}}]}}'
  sleep 10
  echo "after:"; kubectl get pod $P -n $NS -o jsonpath='{.spec.containers[?(@.name=="server")].resources}{"\n"}restarts={.status.containerStatuses[?(@.name=="server")].restartCount}{"\n"}'
  echo "SUCCESS = requests now 200m and restarts unchanged"
  ;;
test-keda)
  kubectl delete hpa --all -n $NS >/dev/null 2>&1 || true
  cat <<EOF | kubectl apply -f -
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata: {name: frontend-rps, namespace: $NS}
spec:
  scaleTargetRef: {name: frontend}
  minReplicaCount: 1
  maxReplicaCount: 5
  pollingInterval: 15
  triggers:
  - type: prometheus
    metadata:
      serverAddress: $PROM
      query: sum(rate(request_total{namespace="default",deployment="frontend",direction="inbound"}[1m]))
      threshold: "20"
EOF
  sleep 20; kubectl get scaledobject,hpa -n $NS
  echo "Now send load (short Locust run); frontend replicas should rise. Remove afterwards with:"
  echo "  kubectl delete scaledobject frontend-rps -n $NS"
  ;;
*) sed -n '2,12p' "$0" ;;
esac
