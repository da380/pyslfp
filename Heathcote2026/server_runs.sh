#!/usr/bin/env bash
#
# server_runs.sh
# ==============
# The runs of the revised paper that are too heavy for a laptop, in the
# order they should be made. Each stage writes a log under logs/, named by
# the stage and the time it started.
#
#   ./server_runs.sh data          fetch and reduce the data
#   ./server_runs.sh probe         the posterior expectations alone, for the
#                                  iteration counts and timings at full scale
#   ./server_runs.sh production    every experiment, with its figures
#   ./server_runs.sh cases         the further cases of the paper's tables,
#                                  each kept in a folder of its own
#   ./server_runs.sh truncation    the data and quantities of interest
#                                  against the truncation degree
#
# The resolution, the cores and the other settings are those of
# run_all.sh, set the same way:
#   CPU_BUDGET=48 ./server_runs.sh production
# run_all.sh has the revised defaults: the ice prior modulated
# (ICE_MODULATION=none for the uniform one), WMB's estimate filtered with
# DDK5 (WMB_FILTER=gaussian for the Gaussian filter), GRACE's block exact
# (GRACE_RANK=k for a sketch of rank k).
#
# RESULTS is where `production` and `cases` copy their outputs, a folder for
# each run with its log (./results if not given): each script writes to
# output_plots_<script>/ and overwrites what is there.
#
# Before the first run:
# - en4_steric.py needs the packages gsw and netCDF4 (pip install gsw
#   netCDF4), which pyslfp does not.
# - The error covariance of GRACE's coefficients is built from 24 monthly
#   files of 0.44 GB each. To skip the download, copy from a machine that
#   has it the two files of ~/.pyslfp_data/itsg_grace2018/reduced/.
# - The statistics of the GIA correction are 1.3 GB of text; pandas, if
#   installed, reads them in seconds. To skip the download, copy
#   ~/.pyslfp_data/caron_gia/reduced/.
#
# On a machine whose home directory is a network share, keep the Python
# environment and the data on a local disk:
#   export PYTHON=/local/path/to/venv/bin/python
#   export PYSLFP_DATA=/local/path/to/pyslfp_data
# Every worker process imports its packages when it starts, and many doing
# so over a share at once spend their time there and not on the work. The
# library and the scripts read their data from PYSLFP_DATA
# (~/.pyslfp_data if not set).

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
export PYTHON
RESULTS="${RESULTS:-results}"
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
    "$PYTHON" ddk_filter.py fetch --levels 1 2 3 4 5 6 7 8
    "$PYTHON" ddk_filter.py check
    "$PYTHON" gia_statistics.py build
    "$PYTHON" gia_statistics.py check
    # The ensemble of GIA models is wanted only with GIA_MODEL=ensemble:
    #   "$PYTHON" gia_ensemble.py --lmax 255 --parallel --max-jobs <cores>
}

# keep <folder> <script, or all>: run_all.sh for the script, with the
# settings of the environment, and its outputs and log copied to
# $RESULTS/<folder>.
keep() {
    local name="$1" script="$2"
    mkdir -p "$RESULTS/$name"
    echo ">>> $name ($(date '+%H:%M:%S'))"
    ./run_all.sh "$script" >"$RESULTS/$name/run.log" 2>&1
    if [ "$script" = all ]; then
        cp -r output_plots_grace_bias output_plots_altimetry_bias \
            output_plots_grace_inversion output_plots_grace_coverage \
            output_plots_altimetry_inversion output_plots_joint_inversion \
            "$RESULTS/$name/"
    else
        cp -r "output_plots_$script" "$RESULTS/$name/"
    fi
}

# The further cases. The middle ocean case (partly) is the default, and is
# in the production run; the first of these makes the maps of the paper's
# load of the Earth System Model with the ocean removed.
cases() {
    OCEAN_CASE=removed keep coverage-ocean-removed grace_coverage
    OCEAN_CASE=full keep coverage-ocean-full grace_coverage
    OCEAN_CASE=removed TRUTH_MEAN=ice-removed \
        keep coverage-ocean-removed-ice-mean-removed grace_coverage
    OCEAN_CASE=removed WMB_FILTER=gaussian \
        keep coverage-ocean-removed-wmb-gaussian grace_coverage
    OCEAN_CASE=removed ICE_MODULATION=none \
        keep coverage-ocean-removed-ice-uniform grace_coverage
    OCEAN_CASE=removed PRIOR_TRUTHS=100 \
        keep coverage-ocean-removed-prior-truths grace_coverage
    OCEAN_CASE=partly PRIOR_TRUTHS=100 \
        keep coverage-ocean-partly-prior-truths grace_coverage
    OCEAN_CASE=removed keep inversion-ocean-removed grace_inversion
    OCEAN_CASE=full keep inversion-ocean-full grace_inversion
    # How much of the errors with the ocean in full is a wrong degree one.
    OCEAN_CASE=full LESS_DEGREE_ONE=1 \
        keep coverage-ocean-full-less-degree-one grace_coverage
    OCEAN_CASE=full PRIOR_TRUTHS=100 \
        keep coverage-ocean-full-prior-truths grace_coverage
}

# The options of the joint experiment as run_all.sh passes them, for
# truncation_check.py: read from the line run_all.sh prints, of a run that
# stops before the script starts.
joint_options() {
    PYTHON=true ./run_all.sh joint_inversion |
        sed -n 's/^>>> args: //p' |
        sed 's/--all//; s/--std-samples [0-9]*//; s/--parallel//; s/--max-jobs [0-9]*//'
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
    # With the joint experiment's test against truths not drawn from the
    # prior and its low-rank posteriors, unless set otherwise.
    export MISMATCH_YEARS="${MISMATCH_YEARS-2007 2020}"
    export LOW_RANK="${LOW_RANK-3000 10000}"
    stage production keep main all
    ;;
cases)
    stage cases cases
    ;;
truncation)
    # shellcheck disable=SC2046
    stage truncation "$PYTHON" truncation_check.py --degrees 63 127 255 511 \
        --samples 3 $(joint_options)
    ;;
*)
    echo "usage: $0 data | probe | production | cases | truncation" >&2
    exit 1
    ;;
esac
