#!/usr/bin/env bash
set -e
mkdir -p perf

# -------- parameters ----------
TAG=${1:-wip}                    # any label: before_batching, gpu_on, etc.
EPISODES=${EPISODES:-20}         # keep it quick & deterministic
OUT="perf/${TAG}_$(date +%Y%m%d_%H%M%S).pstats"
LOG="perf/${TAG}_$(date +%Y%m%d_%H%M%S).log"
# -------------------------------

/usr/bin/time -f "WALL %e s"    \
python -m cProfile -o "$OUT"   \
       src/main.py            \
       --train True           \
       -f src/config/default_cfg.py \
       --cfg_opts             \
           train.num_episodes "$EPISODES" \
           train.gamestate normal_phase  \
           miscs.render_mode none        \
           miscs.verbose False | tee "$LOG"

echo "Profile saved → $OUT"
