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
#   FAST=1 ./run_all.sh                       fast/low-res test run (lmax=64, obs=32, spacing=8)
#   PARALLEL=0 ./run_all.sh                   serial runs (scripts' default)
#   CPU_BUDGET=6 ./run_all.sh                 use at most six cores
#   MAX_JOBS=4 N_THREADS=2 ./run_all.sh       4 workers x 2 threads each
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
# CPU_BUDGET  cores this run may use. Defaults to the physical core count
#             minus one: SMT siblings give nothing for the transforms, and
#             one core is kept free for the desktop and the operating
#             system. Set it explicitly on shared machines.
# N_THREADS   threads per worker process (transforms and BLAS), exported
#             below. 1 is right whenever a call site has at least CPU_BUDGET
#             independent tasks; 2 is the most worth pairing with several
#             workers, since thread efficiency falls off quickly.
# MAX_JOBS    worker-process cap; defaults to CPU_BUDGET / N_THREADS so
#             that MAX_JOBS x N_THREADS <= CPU_BUDGET.
# PARALLEL    1 (default): pass --parallel --max-jobs to every script.
#             0: run the scripts serially (their default).
#
# The main process runs its serial phases with CPU_BUDGET threads
# (--serial-threads). It idles while workers run, so the peak load stays at
# MAX_JOBS x N_THREADS busy threads.
physical_cores() {
    local n=""
    if command -v lscpu >/dev/null 2>&1; then
        n="$(lscpu -p=CORE 2>/dev/null | grep -v '^#' | sort -u | wc -l | tr -d ' ')"
    elif command -v sysctl >/dev/null 2>&1; then
        n="$(sysctl -n hw.physicalcpu 2>/dev/null || true)"
    fi
    if [ -z "$n" ] || [ "$n" -lt 1 ] 2>/dev/null; then
        n="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)"
    fi
    echo "$n"
}

PHYSICAL_CORES="$(physical_cores)"
LOGICAL_CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)"
CPU_BUDGET_DEFAULT=$((PHYSICAL_CORES - 1))
if [ "$CPU_BUDGET_DEFAULT" -lt 1 ]; then CPU_BUDGET_DEFAULT=1; fi
CPU_BUDGET="${CPU_BUDGET:-$CPU_BUDGET_DEFAULT}"
N_THREADS="${N_THREADS:-1}"
PARALLEL="${PARALLEL:-1}"

MAX_JOBS_DEFAULT=$((CPU_BUDGET / N_THREADS))
if [ "$MAX_JOBS_DEFAULT" -lt 1 ]; then MAX_JOBS_DEFAULT=1; fi
MAX_JOBS="${MAX_JOBS:-$MAX_JOBS_DEFAULT}"

if [ $((MAX_JOBS * N_THREADS)) -gt "$CPU_BUDGET" ]; then
    echo "Warning: MAX_JOBS x N_THREADS = $((MAX_JOBS * N_THREADS)) exceeds CPU_BUDGET=$CPU_BUDGET." >&2
fi

export OMP_NUM_THREADS="$N_THREADS"        # OpenMP and ducc0 thread pool in workers
export OPENBLAS_NUM_THREADS="$N_THREADS"
export MKL_NUM_THREADS="$N_THREADS"
export BLIS_NUM_THREADS="$N_THREADS"
export VECLIB_MAXIMUM_THREADS="$N_THREADS" # macOS Accelerate
export NUMEXPR_NUM_THREADS="$N_THREADS"

PAR_ARGS=(--serial-threads "$CPU_BUDGET")
echo "Cores: $PHYSICAL_CORES physical, $LOGICAL_CORES logical; budget $CPU_BUDGET."
if [ "$PARALLEL" -eq 1 ]; then
    PAR_ARGS+=(--parallel --max-jobs "$MAX_JOBS")
    echo "Parallel: up to $MAX_JOBS worker(s) x $N_THREADS thread(s) each; serial phases use $CPU_BUDGET thread(s)."
else
    echo "Serial: $CPU_BUDGET thread(s) in the main process."
fi

PYTHON="${PYTHON:-python}"

# ---------------------------------------------------------------------------
# 3. Resolution / Test mode configuration
# ---------------------------------------------------------------------------
# Toggle quick low-resolution test runs via FAST=1 (or TEST=1):
#   FAST=1 ./run_all.sh
# Individual values can also be explicitly overridden.
FAST="${FAST:-${TEST:-0}}"

if [ "$FAST" -ne 0 ]; then
    LMAX="${LMAX:-64}"
    OBS_DEGREE_VAL="${OBS_DEGREE_VAL:-32}"
    SPACING="${SPACING:-8.0}"
    echo "Resolution: FAST/TEST mode (lmax=$LMAX, obs_degree=$OBS_DEGREE_VAL, spacing=$SPACING)."
else
    LMAX="${LMAX:-256}"
    # The coefficient errors below stop at degree 96; 60 is the default.
    OBS_DEGREE_VAL="${OBS_DEGREE_VAL:-60}"
    SPACING="${SPACING:-1.0}"
fi

# ---------------------------------------------------------------------------
# 4. Arguments
# ---------------------------------------------------------------------------
# Each array below goes only to the scripts that accept every option in it.

# Shared by all five scripts (identical names and defaults everywhere).
COMMON=(
    --lmax "$LMAX"
    --load-scale-km 500.0
    --prior-order 3.0
    --prior-shift 0.0
)

# GRACE observation truncation: grace_bias, grace_inversion, joint_inversion.
OBS_DEGREE=(--obs-degree "$OBS_DEGREE_VAL")

# Preconditioner: altimetry_inversion, joint_inversion only. Blocks
# assembled in the data space (common.block_preconditioner); the surrogate
# problem at --surrogate-degree is the alternative, and what the submitted
# paper used (--preconditioner surrogate).
SURROGATE=(--surrogate-degree 32 --preconditioner blocks)

# Tolerance of the solves that push the posterior covariance onto the
# quantities of interest, the estimator kernels and the covariance maps:
# grace_inversion, altimetry_inversion, joint_inversion. The posterior
# expectation and samples keep the scripts' own tolerance.
QOI=(--qoi-rtol 1e-6)

# Ocean model and prior: altimetry_bias, altimetry_inversion,
# joint_inversion (identical names and defaults in all three). The fields
# lie in H^2; the loads, discontinuous at the supports' edges, in H^(1/4).
# The ice and the water stored on the other land are fields of H^(1/4) with
# priors of order 1.5 on L^2, their amplitudes and scales stated in mm EWT
# and km (from the year-on-year changes of the ESA Earth System Model). The
# submitted paper had the ice alone, in H^2 with the other fields: drop
# --land-water, --source-order and the five options after them for that.
OCEAN_PRIOR=(
    --field-order 2.0
    --load-order 0.25
    --spacing "$SPACING"
    --land-water
    --source-order 0.25
    --source-prior-order 1.5
    --ice-std-mm 80.0
    --ice-scale-km 180.0
    --land-water-std-mm 65.0
    --land-water-scale-km 340.0
    --ice-scale-factor 1.0
    --gmsl-bary-steric-ratio 1.7
    --ocean-dyn-scale-factor 0.2
    --ocean-dyn-std-mm 4.0
    --ocean-rho-scale-factor 1.0
    --steric-dyn-std-ratio 0.75
    --ocean-corr 0.9
    --ocean-corr-scale-factor 0.4
)

# Altimetry noise: the same numbers, but joint_inversion prefixes the flags
# with "alt-".
# For a change over one year, in mm: a local error, a large-scale correlated
# error and its scale in km, an error common to every point, and an error in
# each component of the frame's origin (Ablain et al., 2019; Prandi et al.,
# 2021). The submitted paper had 2, 0.04, 2000, 0 and 0.
ALT_NOISE_STD_MM=6.0
ALT_NOISE_CORR_STD_MM=0.7
ALT_NOISE_CORR_SCALE_KM=2000.0
ALT_NOISE_GLOBAL_STD_MM=1.0
ALT_NOISE_FRAME_STD_MM=0.33

ALT_NOISE=(
    --noise-std-mm "$ALT_NOISE_STD_MM"
    --noise-corr-std-mm "$ALT_NOISE_CORR_STD_MM"
    --noise-corr-scale-km "$ALT_NOISE_CORR_SCALE_KM"
    --noise-global-std-mm "$ALT_NOISE_GLOBAL_STD_MM"
    --noise-frame-std-mm "$ALT_NOISE_FRAME_STD_MM"
)
JOINT_ALT_NOISE=(
    --alt-noise-std-mm "$ALT_NOISE_STD_MM"
    --alt-noise-corr-std-mm "$ALT_NOISE_CORR_STD_MM"
    --alt-noise-corr-scale-km "$ALT_NOISE_CORR_SCALE_KM"
    --alt-noise-global-std-mm "$ALT_NOISE_GLOBAL_STD_MM"
    --alt-noise-frame-std-mm "$ALT_NOISE_FRAME_STD_MM"
)

# The error of the GIA correction, in the altimetry and in GRACE's data of
# the ocean experiments: the spread of the ensemble of gia_ensemble.py over
# one year, scaled. Only the test ensemble exists so far.
GIA=(
    --gia-years 1.0
    --gia-scale 0.4
)

# GRACE noise: the formal error covariance of the coefficients of
# ITSG-Grace2018 for the difference of the annual means of two years
# (grace_covariance.py build --years 2007 2008), a full matrix. The
# submitted paper had a spatial noise field solved from two signal-to-noise
# conditions: for that, drop these and pass --noise-scale-factor 0.5
# --snr-low-degree 10.0 --snr-crossover-degree 64 to the GRACE scripts and
# --grace-noise-scale-km 250.0 --grace-snr-low-degree 10.0
# --grace-snr-crossover-degree 64 to joint_inversion, with OBS_DEGREE_VAL=100.
GRACE_NOISE=(--coefficient-errors 2007 2008)
JOINT_GRACE_NOISE=(--grace-coefficient-errors 2007 2008)

# GRACE prior: grace_bias, grace_inversion. The model is the direct load, in
# H^2, with one Sobolev prior over the globe: still that of the submitted
# paper, not the three fields of grace_heldout.py and grace_coverage.py.
# Optional: --smoothing-scale-km <km> [defaults to --load-scale-km],
# --remove-degree-1.
GRACE_PRIOR=(
    --load-order 2.0
    --direct-scale-km 250
    --direct-std-m 0.01
)

# Per-script extras (plot selections, MC sample counts) -- edit to taste.
# The ${arr[@]+...} expansion below keeps `set -u` happy if you empty one
# of these arrays (needed on bash < 4.4, e.g. macOS /bin/bash).
ALTIMETRY_BIAS_EXTRA=(--plot-maps)  # e.g. --samples 1000
GRACE_BIAS_EXTRA=(--plot-maps)      # e.g. --samples 1000
ALTIMETRY_INV_EXTRA=(--all)         # e.g. --std-samples 100
GRACE_INV_EXTRA=(--all --std-samples 100) # e.g. --prior-sensitivity
JOINT_INV_EXTRA=(--all --std-samples 100) # e.g. --map-all-cases

# ---------------------------------------------------------------------------
# 5. Run
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
        run "$s" "${COMMON[@]}" "${OBS_DEGREE[@]}" "${GRACE_PRIOR[@]}" "${GRACE_NOISE[@]}" \
            ${PAR_ARGS[@]+"${PAR_ARGS[@]}"} \
            ${GRACE_BIAS_EXTRA[@]+"${GRACE_BIAS_EXTRA[@]}"}
        ;;
    altimetry_bias)
        run "$s" "${COMMON[@]}" "${OCEAN_PRIOR[@]}" "${ALT_NOISE[@]}" "${GIA[@]}" \
            ${PAR_ARGS[@]+"${PAR_ARGS[@]}"} \
            ${ALTIMETRY_BIAS_EXTRA[@]+"${ALTIMETRY_BIAS_EXTRA[@]}"}
        ;;
    grace_inversion)
        run "$s" "${COMMON[@]}" "${OBS_DEGREE[@]}" "${GRACE_PRIOR[@]}" "${GRACE_NOISE[@]}" "${QOI[@]}" \
            ${PAR_ARGS[@]+"${PAR_ARGS[@]}"} \
            ${GRACE_INV_EXTRA[@]+"${GRACE_INV_EXTRA[@]}"}
        ;;
    altimetry_inversion)
        run "$s" "${COMMON[@]}" "${SURROGATE[@]}" "${OCEAN_PRIOR[@]}" "${ALT_NOISE[@]}" "${GIA[@]}" "${QOI[@]}" \
            ${PAR_ARGS[@]+"${PAR_ARGS[@]}"} \
            ${ALTIMETRY_INV_EXTRA[@]+"${ALTIMETRY_INV_EXTRA[@]}"}
        ;;
    joint_inversion)
        run "$s" "${COMMON[@]}" "${SURROGATE[@]}" "${OBS_DEGREE[@]}" \
            "${OCEAN_PRIOR[@]}" "${JOINT_ALT_NOISE[@]}" "${JOINT_GRACE_NOISE[@]}" "${GIA[@]}" "${QOI[@]}" \
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
