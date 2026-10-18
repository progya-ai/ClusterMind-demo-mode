#!/usr/bin/env bash
# Mac-side wrapper (first run builds the image, ~5-10 min)
#   ./run.sh <command> [args]   console mode, e.g. ./run.sh demo, ./run.sh tables
#   ./run.sh web                web GUI at http://localhost:8000  (Ctrl-C to stop)
#   ./run.sh web -d             web GUI in the background;  ./run.sh web-stop  to stop it
cd "$(dirname "$0")"
case "${1:-}" in
  web)
    shift
    port="${CM_WEB_PORT:-8000}"
    if [ "${1:-}" = "-d" ]; then
      docker compose up -d web && echo "ClusterMind web GUI: http://localhost:${port}/  (stop: ./run.sh web-stop)"
    else
      echo "ClusterMind web GUI: http://localhost:${port}/  (Ctrl-C to stop)"
      exec docker compose up web
    fi ;;
  web-stop)
    exec docker compose stop web ;;
  *)
    exec docker compose run --rm clustermind "$@" ;;
esac
