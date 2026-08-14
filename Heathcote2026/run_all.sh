#!/usr/bin/env bash
#
# run_all.sh
# ==========
# Runs the bias/inversion scripts with a single consistent set of
# arguments, limiting BLAS/OpenMP threading via environment variables and
# driving process-level parallelism through each script's --parallel /
# --max-jobs options.
#
# The threading variables must be exported BEFORE Python starts: each
# runtime (OpenBLAS, MKL, libgomp, ...) reads them at library load time,
# and worker processes inherit them.
#
# Usage:
#   ./run_all.sh                              run the default set
#   ./run_all.sh all                          run everything
#   ./run_all.sh list                         show the script sets and exit
#   ./run_all.sh grace_bias joint_inversion   run exactly these, in order
#   PARALLEL=0 ./run_all.sh                   serial runs (scripts' default)
#   MAX_JOBS=8 N_THREADS=2 ./run_all.sh       8 workers x 2 threads each
#   PYTHON=python3.12 ./run_all.sh            choose the interpreter

set -euo pipefail
cd "$(dirname "$0")" # the .py scripts are assumed to sit next to this file

# ---------------------------------------------------------------------------
# 1. Script selection
# ---------------------------------------------------------------------------
# altimetry_inversion is omitted from full paper-reconstruction runs (the
# joint script covers the single-sensor comparison); include it by naming
# it explicitly or with "all".
DEFAULT_SCRIPTS=(grace_bias altimetry_bias grace_inversion joint_inversion)
ALL_SCRIPTS=(grace_bias altimetry_bias grace_inversion altimetry_inversion joint_inversion)

if [ $# -eq 0 ]; then
    SCRIPTS=("${DEFAULT_SCRIPTS[@]}")
elif [ "$1" = "all" ]; then
    SCRIPTS=("${ALL_SCRIPTS[@]}")
elif [ "$1" = "list" ]; then
    echo "default: ${DEFAULT_SCRIPTS[*]}"
    echo "all:     ${ALL_SCRIPTS[*]}"
    exit 0
else
    SCRIPTS=("$@")
fi

# ---------------------------------------------------------------------------
# 2. Threading and parallelisation
# ---------------------------------------------------------------------------
# N_THREADS  threads per process (BLAS/OpenMP), exported below.
# PARALLEL   1 (default): pass --parallel --max-jobs to every script.
#            0: pass nothing, so the scripts run serially (their default).
# MAX_JOBS   worker-process cap; defaults to cores / N_THREADS so that
#            n_jobs x N_THREADS <= available cores.
N_THREADS="${N_THREADS:-1}"
CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)"
PARALLEL="${PARALLEL:-1}"

MAX_JOBS_DEFAULT=$((CORES / N_THREADS))
if [ "$MAX_JOBS_DEFAULT" -lt 1 ]; then MAX_JOBS_DEFAULT=1; fi
MAX_JOBS="${MAX_JOBS:-$MAX_JOBS_DEFAULT}"

export OMP_NUM_THREADS="$N_THREADS"        # OpenMP: SHTOOLS, libgomp/libiomp
export OPENBLAS_NUM_THREADS="$N_THREADS"
export MKL_NUM_THREADS="$N_THREADS"
export BLIS_NUM_THREADS="$N_THREADS"
export VECLIB_MAXIMUM_THREADS="$N_THREADS" # macOS Accelerate
export NUMEXPR_NUM_THREADS="$N_THREADS"

PAR_ARGS=()
if [ "$PARALLEL" -eq 1 ]; then
    PAR_ARGS=(--parallel --max-jobs "$MAX_JOBS")
    echo "Parallel: up to $MAX_JOBS worker(s) x $N_THREADS thread(s) each ($CORES cores detected)."
else
    echo "Serial: backend threading limited to $N_THREADS thread(s)."
fi

PYTHON="${PYTHON:-python}"

# ---------------------------------------------------------------------------
# 3. Common arguments
# ---------------------------------------------------------------------------

# Shared by all five scripts (identical names and defaults everywhere).
COMMON=(
    --lmax 256
    --load-order 2.0
    --load-scale-km 500.0
    --prior-kernel sobolev
    --prior-order 1.0
    --prior-shift 1.0
)

# GRACE observation truncation: grace_bias, grace_inversion, joint_inversion.
OBS_DEGREE=(--obs-degree 100)

# Preconditioner truncation: altimetry_inversion, joint_inversion only.
SURROGATE=(--surrogate-degree 32)

# Altimetry-side model priors: altimetry_bias, altimetry_inversion,
# joint_inversion (identical names and defaults in all three).
#
# One dimensioned amplitude anchors the chain: the pointwise
# sterodynamic std (mm, pre-mass-constraint; ~ the observed regional
# trend spread for per-year fields). The density amplitude follows from
# the steric/sterodynamic std ratio, and the ice amplitude from the
# barystatic/steric ratio of the REALISED GMSL prior stds under the
# mass constraint. Derived stds and realised GMSL statistics are 
# printed by each run.
ALT_PRIOR=(
    --spacing 1.0
    --ice-scale-factor 1.0
    --gmsl-bary-steric-ratio 1.7
    --ocean-dyn-scale-factor 0.2
    --ocean-dyn-std-mm 4.0
    --ocean-rho-scale-factor 1.0
    --steric-dyn-std-ratio 0.75
    --ocean-corr 0.9
    --ocean-corr-scale-factor 0.4
)

# Altimetry noise: same numbers, but joint_inversion prefixes the flags
# with "alt-". Single source of truth here keeps the runs consistent.
# Factors of the sterodynamic pointwise prior std.
ALT_NOISE_STD=0.5
ALT_NOISE_CORR_STD=0.025
ALT_NOISE_CORR_SCALE=4.0

ALT_NOISE=(
    --noise-std-factor "$ALT_NOISE_STD"
    --noise-corr-std-factor "$ALT_NOISE_CORR_STD"
    --noise-corr-scale-factor "$ALT_NOISE_CORR_SCALE"
)
JOINT_ALT_NOISE=(
    --alt-noise-std-factor "$ALT_NOISE_STD"
    --alt-noise-corr-std-factor "$ALT_NOISE_CORR_STD"
    --alt-noise-corr-scale-factor "$ALT_NOISE_CORR_SCALE"
)

# GRACE-family prior/noise parameterisation: grace_bias, grace_inversion.
# (--noise-std-factor here is a different quantity from the altimetry one,
# which is why it lives in this array and not a shared one.  Optional:
# --smoothing-scale-km <km> [defaults to --load-scale-km], --remove-degree-1.)
GRACE_PRIOR=(
    --direct-scale-km 250.0
    --direct-std-m 0.01
    --noise-scale-factor 0.25
    --noise-std-factor 0.1414
)

# joint_inversion's own GRACE noise parameterisation (deliberately
# independent of GRACE_PRIOR above); the std is a factor of the derived
# ice pointwise std, the dominant load.
JOINT_GRACE_NOISE=(
    --grace-noise-scale-km 50.0
    --grace-noise-std-factor 0.1
)

# Per-script extras (plot selections, MC sample counts) -- edit to taste.
# The ${arr[@]+...} expansion below keeps `set -u` happy if you empty one
# of these arrays (needed on bash < 4.4, e.g. macOS /bin/bash).
ALTIMETRY_BIAS_EXTRA=(--plot-maps)  # e.g. --samples 1000
GRACE_BIAS_EXTRA=(--plot-maps)      # e.g. --samples 1000
ALTIMETRY_INV_EXTRA=(--all)         # e.g. --std-samples 100
GRACE_INV_EXTRA=(--all)             # e.g. --prior-sensitivity
JOINT_INV_EXTRA=(--all --std-samples 100)             # e.g. --map-all-cases --std-samples 100

# ---------------------------------------------------------------------------
# 4. Run
# ---------------------------------------------------------------------------
run() {
    local name="$1"
    shift
    echo
    echo "======================================================================"
    echo ">>> $name.py  ($(date '+%H:%M:%S'))"
    echo ">>> args: $*"
    echo "======================================================================"
    time "$PYTHON" "$name.py" "$@"
}

for s in "${SCRIPTS[@]}"; do
    case "$s" in
    grace_bias)
        run "$s" "${COMMON[@]}" "${OBS_DEGREE[@]}" "${GRACE_PRIOR[@]}" \
            ${PAR_ARGS[@]+"${PAR_ARGS[@]}"} \
            ${GRACE_BIAS_EXTRA[@]+"${GRACE_BIAS_EXTRA[@]}"}
        ;;
    altimetry_bias)
        run "$s" "${COMMON[@]}" "${ALT_PRIOR[@]}" "${ALT_NOISE[@]}" \
            ${PAR_ARGS[@]+"${PAR_ARGS[@]}"} \
            ${ALTIMETRY_BIAS_EXTRA[@]+"${ALTIMETRY_BIAS_EXTRA[@]}"}
        ;;
    grace_inversion)
        run "$s" "${COMMON[@]}" "${OBS_DEGREE[@]}" "${GRACE_PRIOR[@]}" \
            ${PAR_ARGS[@]+"${PAR_ARGS[@]}"} \
            ${GRACE_INV_EXTRA[@]+"${GRACE_INV_EXTRA[@]}"}
        ;;
    altimetry_inversion)
        run "$s" "${COMMON[@]}" "${SURROGATE[@]}" "${ALT_PRIOR[@]}" "${ALT_NOISE[@]}" \
            ${PAR_ARGS[@]+"${PAR_ARGS[@]}"} \
            ${ALTIMETRY_INV_EXTRA[@]+"${ALTIMETRY_INV_EXTRA[@]}"}
        ;;
    joint_inversion)
        run "$s" "${COMMON[@]}" "${SURROGATE[@]}" "${OBS_DEGREE[@]}" \
            "${ALT_PRIOR[@]}" "${JOINT_ALT_NOISE[@]}" "${JOINT_GRACE_NOISE[@]}" \
            ${PAR_ARGS[@]+"${PAR_ARGS[@]}"} \
            ${JOINT_INV_EXTRA[@]+"${JOINT_INV_EXTRA[@]}"}
        ;;
    *)
        echo "Unknown script: $s" >&2
        echo "Choose from: ${ALL_SCRIPTS[*]} (or the keywords 'all', 'list')" >&2
        exit 1
        ;;
    esac
done

echo
echo "All done."