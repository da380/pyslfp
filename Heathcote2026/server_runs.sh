#!/usr/bin/env bash
#
# server_runs.sh
# ==============
# The runs of the revised paper that are too heavy for a laptop, in the
# order they should be made. Each stage writes a log under logs/, named by
# the stage and the time it started.
#
#   ./server_runs.sh data          fetch and reduce the data, build the GIA
#                                  ensemble at the runs' degree
#   ./server_runs.sh probe         the posterior expectations alone, for the
#                                  iteration counts and timings at full scale
#   ./server_runs.sh production    every experiment, with its figures
#
# The resolution and the cores are those of run_all.sh, set the same way:
#   LMAX=256 SPACING=1.0 CPU_BUDGET=32 ./server_runs.sh probe
#
# Before the first run:
# - en4_steric.py needs the packages gsw and netCDF4 (pip install gsw
#   netCDF4), which pyslfp does not.
# - The error covariance of GRACE's coefficients is built from 24 monthly
#   files of 0.44 GB each. To skip the download, copy from a machine that
#   has it the two files of ~/.pyslfp_data/itsg_grace2018/reduced/.
# - The ocean prior is still to be decided: make the probe only once it is,
#   since the size of the ocean signal sets the altimetry's information.

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
LMAX="${LMAX:-256}"
export LMAX PYTHON
mkdir -p logs
stamp="$(date '+%Y%m%d-%H%M%S')"

stage() {
    local name="$1"
    shift
    echo ">>> $name ($(date '+%H:%M:%S')); log: logs/$name-$stamp.log"
    "$@" 2>&1 | tee "logs/$name-$stamp.log"
}

data() {
    "$PYTHON" grace_covariance.py build --years 2007 2008
    "$PYTHON" esm_loads.py fetch
    "$PYTHON" en4_steric.py fetch
    "$PYTHON" ddk_filter.py fetch --levels 3 5
    "$PYTHON" ddk_filter.py check
    # 108 members; each is independent of the others.
    "$PYTHON" gia_ensemble.py --lmax "$LMAX" --parallel \
        --max-jobs "${GIA_JOBS:-${CPU_BUDGET:-8}}"
}

case "${1:-}" in
data)
    stage data data
    ;;
probe)
    PROBE=1 stage probe ./run_all.sh grace_inversion altimetry_inversion joint_inversion
    grep -n ">>> [a-z]\|iterations\.\|Squared misfit\|^real" "logs/probe-$stamp.log"
    ;;
production)
    stage production ./run_all.sh all
    ;;
*)
    echo "usage: $0 data | probe | production" >&2
    exit 1
    ;;
esac
