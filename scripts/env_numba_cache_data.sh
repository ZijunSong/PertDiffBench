# Source before running PertDiffBench jobs so Numba JIT cache lives on /data/ppnm (avoids filling $HOME).
# Usage: source /data/ppnm/PertDiffBench/scripts/env_numba_cache_data.sh
# Or:    bash -c 'source ... && nohup bash scripts/... &'
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/data/ppnm/cache/numba}"
mkdir -p "$NUMBA_CACHE_DIR"
# Manual scDiff: if dcor hits LLVM "Symbol not found", use NUMBA_DISABLE_JIT=1 (fig2_task2_plus_scdiff.sh sets it by default).
