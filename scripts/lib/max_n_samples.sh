# Shared helpers for max evaluation n_samples (source from experiment shell scripts).
# Usage:
#   source "$(dirname "$0")/../lib/max_n_samples.sh"   # from scripts/fig*/
#   N_SAMPLES="$(max_n_samples_paired "${TEST_H5AD}")"

max_n_samples_paired() {
  local h5ad="$1"
  local mode="${2:-paired_ifn}"
  python "${PERTDIFFBENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/tools/max_n_samples_from_h5ad.py" \
    "${h5ad}" --mode "${mode}"
}

max_n_samples_multi_pert() {
  max_n_samples_paired "$1" "multi_pert"
}

max_n_samples_timepoint() {
  max_n_samples_paired "$1" "timepoint"
}

# Fill SAMPLES_MAP[cell_type] from valid h5ad paths for fig1 task1-style loops.
# Args: ROOT_DIR  CELL_TYPES...
build_samples_map_from_valid_h5ad() {
  local root_dir="$1"
  shift
  local cell_type
  for cell_type in "$@"; do
    local valid_path="${root_dir}data/highly_variable_gene_gradient/${cell_type}_valid_HVG_1000.h5ad"
    if [[ ! -f "${valid_path}" ]]; then
      echo "ERROR: missing valid h5ad for ${cell_type}: ${valid_path}" >&2
      return 1
    fi
    SAMPLES_MAP["${cell_type}"]="$(max_n_samples_paired "${valid_path}")"
  done
}
