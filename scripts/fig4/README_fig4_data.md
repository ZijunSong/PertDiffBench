# Fig4 时间条件生成任务 — 数据处理说明

## 1. 原始数据

目录：`/data/ppnm/data/PertDiffBench/data_ori/fig4/`（与仓库内 `data_ori/fig4` 对应）

- **表达矩阵**: `GSM3770930_A549_lognorm_scale_hvg3000.csv`（细胞×基因，已 log2 标准化，3000 HVG）
- **元数据**: `GSM3770930_A549_cell_annotate.txt`，列含 `sample`, `treatment_time`, `doublet_score` 等
- **时间点**: 0h, 2h, 4h, 6h, 8h, 10h（各时间点细胞数约 1k–1.4k）

## 2. 是否用 perturbation_status，以及如何设

- 本任务**没有**真实的 Control/IFN 分组，只有 `treatment_time`。
- 为兼容现有依赖 `perturbation_status` 的代码（如 scDiffusion 的 classifier 若仍从该列读二分类），可以**仅在训练集**里做如下**约定**：
  - **训练集**：`0h, 2h` → `perturbation_status = "Control"`；`8h, 10h` → `perturbation_status = "IFN"`。
- 这样做的含义是：「早期时间点」视为 Control，「晚期时间点」视为 IFN，仅用于满足接口或二分类兼容；**真正用于时间条件生成的是 `treatment_time` 列**（0h/2h/8h/10h 作为多类标签）。
- **测试集**（4h, 6h）不参与训练，`perturbation_status` 可统一设为 `"IFN"` 或 `"Holdout"`，仅作占位；评估时以 `treatment_time` 区分 4h 与 6h 即可。

**结论**：  
- 训练集中把 0h/2h 标为 Control、8h/10h 标为 IFN 是合理且便于兼容的。  
- 测试集不需要“真实”的 Control/IFN，只要保留 `treatment_time` 用于评估即可。

## 3. 数据划分（训练 / 测试）

- **训练集 (train)**  
  - 只使用 **0h, 2h, 8h, 10h** 的细胞。  
  - 用于训练时间条件生成模型（如 scDiffusion 的 diffusion + classifier，条件为时间点）。

- **测试集 (test)**  
  - 只使用 **4h, 6h** 的细胞。  
  - 用于评估：在推理时给定时间标签 4h、6h 分别生成细胞，与真实 4h、6h 细胞比较（MMD、LISI 等）。

**不必**在 0h/2h/8h/10h 中再拆一份做“验证集”，除非你希望做 early stopping；若做，可从训练时间点中随机留约 10% 作为 valid。

## 4. 建议的 h5ad 产出

输出目录：`/data/ppnm/data/PertDiffBench/data/fig4_task1/`（由 `preprocess_data/fig4/prepare_fig4_h5ad.py` 生成）

| 文件 | 内容 | 用途 |
|------|------|------|
| `fig4_train.h5ad` | 0h + 2h + 8h + 10h 细胞 | 训练（VAE + Diffusion + Classifier 等） |
| `fig4_test.h5ad`  | 4h + 6h 细胞 | 评估生成质量（与真实 4h/6h 对比） |

可选：  
- `fig4_valid.h5ad`：从 0h/2h/8h/10h 中留约 10% 作验证（可选）。

## 5. obs 列建议

- **必须保留**  
  - `treatment_time`：字符串，如 `"0h"`, `"2h"`, `"4h"`, `"6h"`, `"8h"`, `"10h"`。  
  - 用于：时间条件训练的标签、评估时按时间点分组。

- **兼容现有 pipeline**  
  - `perturbation_status`：  
    - 训练集：0h/2h → `"Control"`，8h/10h → `"IFN"`。  
    - 测试集：可全部设为 `"IFN"` 或 `"Holdout"`。

- **可选**  
  - `split`：`"train"` / `"test"`（或 `"valid"`），便于检查划分。  
  - 其他元数据（如 `doublet_score`）按需保留。

## 6. 与 fig1 的差异（简要）

- fig1：每个细胞有真实 Control/IFN 标签，train/valid 按细胞类型或随机划分，且通常 Control 与 IFN 成对或同细胞类型。  
- fig4：无 Control/IFN，只有时间；**按时间点划分**：训练时间点 (0h,2h,8h,10h) → train，未见时间点 (4h,6h) → test；`perturbation_status` 仅在训练集按上面约定填写，以兼容现有代码。

## 7. 脚本与评估（scripts/fig4）

- **数据处理**：`preprocess_data/fig4/prepare_fig4_h5ad.py` 从 `data_ori/fig4` 读入 CSV/元数据，生成上述 `fig4_train.h5ad` 与 `fig4_test.h5ad`。  
- **时间条件评估**：`scripts/fig4/eval_fig4_time_conditioned.py` 按 `treatment_time` 分组计算 11 项指标（与 fig1 一致），需 `--test-h5ad`、`--generated-h5ad`，可选 `--train-h5ad`（0h 作 control 算 delta 类指标）。  
- **baseline 脚本**：`fig4_task1_scdiffusion.sh`、`fig4_task1_ddpm.sh`、`fig4_task1_ddpm_mlp.sh`、`fig4_task1_scdiff.sh`、`fig4_task1_squidiff.sh` 仿 fig1 逻辑：对 fig4 数据做多轮训练+测评并汇总 CSV。  
- **无 Control/IFN 的 test**：fig4_test 仅含 4h/6h，无 Control。若 baseline 的 eval 要求 test 中同时有 Control 与 IFN，则使用 **时间条件模式**：`--time-conditioned` + `--generated-h5ad`，由 `eval_fig4_time_conditioned.py` 按时间分组评估。

## 8. 各方法如何生成 4h/6h（与图2 对齐）

| 方法 | 4h/6h 生成方式 |
|------|----------------|
| **DDPM** | 为 fig4 单独训练一个 VAE（仅 encoder+decoder，与 DDPM+MLP 同结构），脚本：`train_fig4_ae_for_ddpm.py`；再用该 VAE 做 2h/8h 线性插值生成 4h/6h，脚本：`sample_fig4_vae_linear_interp.py`。 |
| **DDPM+MLP** | 使用本模型自带的 encoder/decoder 做 2h/8h 线性插值 → 4h/6h（不跑 diffusion），脚本：`sample_fig4_vae_linear_interp.py`，ckpt 为 DDPM+MLP 的 `model_epoch_1000.pth`。 |
| **Squidiff** | 在 Squidiff latent 空间对 2h/8h 做线性插值得到 4h/6h latent，再经 diffusion 解码为表达。脚本：`sample_fig4_squidiff_interp.py`。 |
| **scDiffusion** | 使用 classifier 的 **梯度插值**（2h–8h 方向）生成 4h/6h，不改为线性插值。 |
