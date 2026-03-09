# State + DDPM Encoder Experiment

使用 [State](https://github.com/ArcInstitute/state) 的 **State Embedding (SE)** 模型作为 encoder，配合 DDPM 实现扰动预测：

```
扰动前 scRNA -> State SE encoder -> DDPM -> 扰动后 scRNA
```

## 环境准备

### 1. 安装 State

```bash
uv tool install arc-state
```

或从源码安装：
```bash
git clone https://github.com/ArcInstitute/state.git
cd state
uv tool install -e .
```

### 2. 下载 SE-600M 预训练模型

从 [HuggingFace](https://huggingface.co/arcinstitute/SE-600M) 下载模型到本地，例如：

```bash
# 使用 huggingface-cli
huggingface-cli download arcInstitute/SE-600M --local-dir /path/to/SE-600M
```

目录结构应包含：
- `se600m_epoch15.ckpt` 或其他 `.ckpt` 文件
- 模型相关配置文件

## 运行

在 PertBench 根目录下执行：

```bash
cd /share/PertBench

# 设置模型路径（可选，默认 checkpoints/SE-600M）
export STATE_MODEL_DIR=/path/to/SE-600M
export STATE_CHECKPOINT=/path/to/SE-600M/se600m_epoch15.ckpt  # 可选，不设则自动选择 .ckpt

bash scripts/encoder_exp/state/state_ddpm.sh
```

或直接修改 `state_ddpm.sh` 中的 `STATE_MODEL_DIR` 和 `STATE_CHECKPOINT` 变量。

## 输出

- 编码结果：`samples/encoder_exp/state_ddpm/task1_train_CD4T_with_state_latent.h5ad` 等
- DDPM  checkpoint：`checkpoints/state_ddpm/latent_ddpm/run_1/` 等
- 评估 metrics：`samples/encoder_exp/state_ddpm/metrics_CD4T.csv`
