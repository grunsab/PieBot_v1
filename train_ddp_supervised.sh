#!/usr/bin/env bash
set -euo pipefail

# train_ddp_supervised.sh
# Supervised pretrain launcher (DDP) with recommended flags:
#   --mode supervised --input-planes 112 --use-rope --use-alibi --mixed-precision
# Usage examples:
#   ./train_ddp_supervised.sh --ccrl-dir /abs/path/to/reformatted --nproc 3 --batch-size-total 384
#   NPROC=4 BATCH_SIZE_TOTAL=512 ./train_ddp_supervised.sh
#   LOGFILE=run.out PIDFILE=run.pid ./train_ddp_supervised.sh --background
#   ./train_ddp_supervised.sh -- --epochs 20  # pass extra args to train_luna.py after --

# Defaults (override via env or flags below)
NPROC="${NPROC:-3}"
RDZV_BACKEND="${RDZV_BACKEND:-c10d}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:29500}"
BATCH_SIZE_TOTAL="${BATCH_SIZE_TOTAL:-384}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${REPO_DIR}/train_luna.py"
CCRL_DIR_DEFAULT="${REPO_DIR}/games_training_data/reformatted"
CCRL_DIR="${CCRL_DIR:-$CCRL_DIR_DEFAULT}"
BACKGROUND=0
EXTRA_ARGS=()

# Parse minimal flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ccrl-dir)
      CCRL_DIR="$2"; shift 2;;
    --nproc)
      NPROC="$2"; shift 2;;
    --batch-size-total)
      BATCH_SIZE_TOTAL="$2"; shift 2;;
    --rdzv)
      RDZV_ENDPOINT="$2"; shift 2;;
    --rdzv-backend)
      RDZV_BACKEND="$2"; shift 2;;
    --background)
      BACKGROUND=1; shift;;
    --)
      shift; EXTRA_ARGS+=("$@"); break;;
    *)
      EXTRA_ARGS+=("$1"); shift;;
  esac
done

# Basic validation
if [[ ! -f "$SCRIPT" ]]; then
  echo "Error: train_luna.py not found at $SCRIPT" >&2
  exit 1
fi
if [[ ! -d "$CCRL_DIR" ]]; then
  echo "Error: CCRL directory not found: $CCRL_DIR" >&2
  echo "Set CCRL_DIR env or pass --ccrl-dir /abs/path/to/reformatted" >&2
  exit 1
fi

CMD=(python3 -m torch.distributed.run
  --nproc_per_node="${NPROC}"
  --rdzv_backend="${RDZV_BACKEND}"
  --rdzv_endpoint="${RDZV_ENDPOINT}"
  "${SCRIPT}"
  --mode supervised
  --input-planes 112
  --use-rope --use-alibi --mixed-precision
  --ccrl-dir "${CCRL_DIR}"
  --distributed
  --batch-size-total "${BATCH_SIZE_TOTAL}")

# Append passthrough args
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Launching: ${CMD[*]}"
if [[ "$BACKGROUND" == "1" ]]; then
  LOGFILE="${LOGFILE:-${REPO_DIR}/luna_train.out}"
  PIDFILE="${PIDFILE:-${REPO_DIR}/luna_train.pid}"
  # Detach so the elastic agent won’t receive SIGHUP on terminal exit
  setsid nohup "${CMD[@]}" > "$LOGFILE" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  echo "Background PID: $(cat "$PIDFILE")"
  echo "Logs: $LOGFILE"
else
  exec "${CMD[@]}"
fi
