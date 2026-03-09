#!/bin/bash
# 检查当前环境是否能让 MindSpore 使用 GPU（无需 sudo）
# 用法: conda activate cellfm 后执行 bash scripts/encoder_exp/cellfm/check_gpu_env.sh

echo "=== 1. 是否在 conda 环境中 ==="
if [ -z "${CONDA_PREFIX}" ]; then
  echo "  未检测到 conda 环境 (CONDA_PREFIX 为空)。请先: conda activate cellfm"
  exit 1
fi
echo "  CONDA_PREFIX=$CONDA_PREFIX"

echo ""
echo "=== 2. 本机是否有 NVIDIA GPU ==="
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || nvidia-smi
else
  echo "  未找到 nvidia-smi，本机可能无 GPU 或驱动未安装。若确定有 GPU，可尝试用 conda 安装 CUDA 库。"
fi

echo ""
echo "=== 3. 查找 libcuda / libcudnn / libcublas ==="
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
    echo "  $lib: 找到 (在 $found)"
  else
    echo "  $lib: 未找到"
  fi
done

echo ""
echo "=== 4. 建议 ==="
if [ -f "${CONDA_PREFIX}/lib/libcudnn.so" ] || [ -f "${CONDA_PREFIX}/lib/libcudnn.so.8" ]; then
  echo "  当前 conda 环境中已有 cuDNN。若仍报错，请运行 pipeline 时确保: conda activate cellfm 后再执行 cellfm_ddpm.sh"
elif [ -d /usr/local/cuda/lib64 ] && [ -f /usr/local/cuda/lib64/libcudnn.so ]; then
  echo "  系统已安装 CUDA/cuDNN。请确认运行前 export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
else
  echo "  建议在 cellfm 环境中用 conda 安装 CUDA 库（无需 sudo）："
  echo "    conda activate cellfm"
  echo "    conda install -y -c nvidia cuda-toolkit=11.8 cudnn"
  echo "  或: mamba install -y -c nvidia cuda-toolkit=11.8 cudnn"
  echo "  安装后重新运行本脚本检查，再运行 cellfm_ddpm.sh"
fi
