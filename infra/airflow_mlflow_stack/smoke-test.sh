#!/usr/bin/env bash
set -u

FAILED=0

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

FRONTEND_PORT="${FRONTEND_PORT:-80}"
BACKEND_PORT="${BACKEND_PORT:-5000}"
PYTHON_DOC_PORT="${PYTHON_DOC_PORT:-8000}"
AIRFLOW_WEB_PORT="${AIRFLOW_WEB_PORT:-8091}"
MLFLOW_PORT="${MLFLOW_PORT:-5001}"
AIRFLOW_ADMIN_USERNAME="${AIRFLOW_ADMIN_USERNAME:-airflow}"
AIRFLOW_ADMIN_PASSWORD="${AIRFLOW_ADMIN_PASSWORD:-change-me}"

log_ok() {
  printf '[OK] %s\n' "$1"
}

log_fail() {
  printf '[FAIL] %s\n' "$1"
  FAILED=1
}

check_http_any() {
  local name="$1"
  local url="$2"
  local code
  code="$(curl -sS -o /dev/null -w "%{http_code}" "$url" || true)"
  if [[ "$code" == "000" ]]; then
    log_fail "$name unreachable ($url)"
  else
    log_ok "$name reachable ($url, status=$code)"
  fi
}

check_http_200() {
  local name="$1"
  local url="$2"
  local code
  code="$(curl -sS -o /dev/null -w "%{http_code}" "$url" || true)"
  if [[ "$code" == "200" ]]; then
    log_ok "$name healthy ($url)"
  else
    log_fail "$name not healthy ($url, status=$code)"
  fi
}

check_airflow_api() {
  local url="http://localhost:${AIRFLOW_WEB_PORT}/api/v2/version"
  local code
  code="$(curl -sS -u "${AIRFLOW_ADMIN_USERNAME}:${AIRFLOW_ADMIN_PASSWORD}" -o /dev/null -w "%{http_code}" "$url" || true)"
  if [[ "$code" == "200" ]]; then
    log_ok "Airflow API healthy ($url)"
  else
    log_fail "Airflow API not healthy ($url, status=$code)"
  fi
}

check_backend_to_python() {
  local output
  output="$(docker compose exec -T backend node -e "fetch('http://python-doc-service:8000/docs').then(r=>{console.log(r.status);process.exit(r.status===200?0:1)}).catch(()=>process.exit(1))" 2>/dev/null || true)"
  if [[ "$output" == "200" ]]; then
    log_ok "Backend -> python-doc-service connectivity"
  else
    log_fail "Backend cannot reach python-doc-service"
  fi
}

printf '== Docker Service Status ==\n'
docker compose ps || true
printf '\n'

check_http_200 "Frontend" "http://localhost:${FRONTEND_PORT}/"
check_http_any "Backend" "http://localhost:${BACKEND_PORT}/"
check_http_200 "Python doc service" "http://localhost:${PYTHON_DOC_PORT}/docs"
check_http_200 "MLflow" "http://localhost:${MLFLOW_PORT}/health"
check_airflow_api
check_backend_to_python

printf '\n'
if [[ "$FAILED" -eq 0 ]]; then
  printf 'All smoke checks passed.\n'
  exit 0
fi

printf 'One or more smoke checks failed.\n'
exit 1
