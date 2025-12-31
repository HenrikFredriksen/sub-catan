#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-smoke}"
STEPS="${2:-200000}"
RUNS="${3:-3}"
THRESH="${4:-1.20}" # fail if new run is 20% slower than baseline
BASELINE_FILE="perf/baseline_${TAG}.txt"
OUTDIR="perf"
mkdir -p "${OUTDIR}"

run_once() {
    local run_id="$1"
    local logfile="${OUTDIR}/${TAG}_${run_id}.log"

    python src/main.py \
        --train \
        --perf \
        --max_env_calls "${STEPS}" \
        --seed 0 \
        --cfg_opts \
            train.num_episodes 1000000 \
            train.gamestate normal_phase \
        2>&1 | tee "${logfile}" >/dev/null

    # extract wall_s from PERF_SUMMARY
    awk '/PERF_SUMMARY/{
        for(i=1;i<=NF;i++){
            if($i ~ /^wall_s=/){split($i,a,"="); print a[2]}
        }
    }' "${logfile}" | tail -n1
}

vals=()
for i in $(seq 1 "${RUNS}"); do
    v=$(run_once "$i")
    echo "run $i: wall_s=$v"
    vals+=("$v")
done

# median
median=$(python - <<PY
import statistics
vals = [float(x) for x in """${vals[*]}""".split()]
print(statistics.median(vals))
PY
)

echo "MEDIAN wall_s=$median"

if [[ -f "$BASELINE_FILE" ]]; then
  base=$(cat "$BASELINE_FILE")
  python - <<PY
base=float("$base"); med=float("$median"); thr=float("$THRESH")
ratio=med/base if base>0 else 0
print(f"baseline={base} median={med} ratio={ratio}")
if base>0 and ratio>thr:
    raise SystemExit(1)
PY
  echo "PASS: no regression (threshold $THRESH)"
else
  echo "$median" > "$BASELINE_FILE"
  echo "Wrote baseline to $BASELINE_FILE"
fi