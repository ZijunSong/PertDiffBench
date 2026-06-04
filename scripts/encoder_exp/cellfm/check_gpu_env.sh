#!/bin/bash
# checkcurrent whether MindSpore using GPU (no need sudo)
# using : conda activate cellfm after bash scripts/encoder_exp/cellfm/check_gpu_env.sh

echo "=== 1. whetherin conda ==="
if [ -z "${CONDA_PREFIX}" ]; then
  echo " to conda (CONDA_PREFIX asempty). : conda activate cellfm"
  exit 1
fi
echo "  CONDA_PREFIX=$CONDA_PREFIX"

echo ""
echo "=== 2. whether NVIDIA GPU ==="
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || nvidia-smi
else
  echo " found nvidia-smi, may GPU or . GPU, can using conda CUDA ."
fi

echo ""
echo "=== 3. libcuda / libcudnn / libcublas ==="
for lib in libcuda.so libcudnn.so libcublas.so; do
  found=""
  for dir in "${CONDA_PREFIX}/lib" /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu; do
    [ -d "$dir" ] || continue
    if ls "$dir"/$lib* 1>/dev/null 2>&1; then
      found="$dir"
      break
    fi
  done
  if [ -n "$found" ]; then
    echo "  $lib: found (in $found)"
  else
    echo " $lib: found"
  fi
done

echo ""
echo "=== 4. recommendation ==="
if [ -f "${CONDA_PREFIX}/lib/libcudnn.so" ] || [ -f "${CONDA_PREFIX}/lib/libcudnn.so.8" ]; then
  echo " current conda cuDNN. , pipeline whenEnsure: conda activate cellfm after cellfm_ddpm.sh"
elif [ -d /usr/local/cuda/lib64 ] && [ -f /usr/local/cuda/lib64/libcudnn.so ]; then
  echo " CUDA/cuDNN. before export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
else
  echo " recommendationin cellfm using conda CUDA (no need sudo): "
  echo "    conda activate cellfm"
  echo "    conda install -y -c nvidia cuda-toolkit=11.8 cudnn"
  echo "  or: mamba install -y -c nvidia cuda-toolkit=11.8 cudnn"
  echo " after check, cellfm_ddpm.sh"
fi
