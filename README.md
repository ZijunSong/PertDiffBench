<div align= "center">
    <h1> 🌊 PertBench: Perturbation Modeling with Diffusion Models Benchmark </h1>
</div>

## ⚙️ Configure the environment and prepare the data

### 🛠️ Configure the environment

```
conda create -n pertbench python=3.10 -y && conda activate pertbench
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 
pip install omegaconf numpy anndata tqdm scanpy gdown einops torch_geometric adjustText wandb 
pip install git+https://github.com/LouiseDck/scgen
sudo apt update
sudo apt install openmpi-bin libopenmpi-dev
pip install mpi4py
```

### 📥 Download the data and the pre-train model



## 📈 Evaluation

### Highly variable gene gradient

In the data of Task 1 in Figure 1, the CD4T cell type has the largest number of cells (5,564), and is therefore chosen as the representative.

First, run `python scripts/tools/get_the_hvg_data_for_fig1.py` to generate the hvg data. Then run

```
nohup bash scripts/highly_variable_gene_gradient/ddpm_hvg.sh > ddpm_hvg.log 2>&1
nohup bash scripts/highly_variable_gene_gradient/ddpm_mlp_hvg.sh > ddpm_mlp_hvg.log 2>&1
nohup bash scripts/highly_variable_gene_gradient/scdiff_hvg.sh > scdiff_hvg.log 2>&1
nohup bash scripts/highly_variable_gene_gradient/scgen_hvg.sh > scgen_hvg.log 2>&1
nohup bash scripts/highly_variable_gene_gradient/squidiff_hvg.sh > squidiff_hvg.log 2>&1
nohup bash scripts/highly_variable_gene_gradient/scdiffusion_hvg.sh > scdiffusion_hvg.log 2>&1
```

to obtain the evaluation results, respectively.

### Fig 1

#### Task 1

**0  Get the data**

Since, overall, the models trained on the data with the lowest number of highly variable genes (1000) achieved the best performance, the experiments of Task 1 and Task 3 in Figure 1 are conducted using the processed data with 1000 HVGs extracted from the original data.  

First, run `python scripts/tools/get_the_hvg_data_for_fig3.py` to generate the data used in the Task 3 experiment of Figure 1. Then, organize this data together with the data obtained from the highly variable gene gradient experiments, for example:

```
/PertBench/
├── /data/
│  ├── /hvg_fig1/
│  │  └── B_train_HVG_1000.h5ad
│  ├── /hvg_fig3/
│  │  └── mix2_test_HVG_1000.h5ad
```

**1  Squidiff**

测试不同高变基因梯度的评估结果

```bash
nohup bash scripts/fig1/fig1_task1_squidff_hvg.sh > fig1_task1_squidff_hvg.log 2>&1
```

选定最佳高变基因数（1000）进行 task1 的全部测评

```bash
nohup bash scripts/fig1/fig1_task1_squidff.sh > fig1_task1_squidff.log 2>&1
```



```
nohup bash scripts/add_gaus/squidiff.sh > add_gaus_squidiff.log 2>&1
```



**2  scDiff**

测试不同高变基因梯度的评估结果

````bash
nohup bash scripts/fig1/fig1_task1_scdiff_hvg.sh > fig1_task1_scdiff_hvg.log 2>&1
````

选定最佳高变基因数（default）进行 task1 的全部测评

```bash
nohup bash scripts/fig1/fig1_task1_scdiff.sh > fig1_task1_scdiff.log 2>&1
```



```
nohup bash scripts/add_gaus/scdiff.sh > add_gaus_scdiff.log 2>&1
```



**3  scDiffusion**

```bash
nohup bash scripts/fig1/fig1_task1_scdiffusion_hvg.sh > fig1_task1_scdiffusion_hvg.log 2>&1
```

6000

```bash
nohup bash scripts/fig1/fig1_task1_scdiffusion.sh > fig1_task1_scdiffusion.log 2>&1
```



```
nohup bash scripts/add_gaus/scdiffusion.sh > add_gaus_scdiffusion.log 2>&1
```



**4  scGen**

测试不同高变基因梯度的评估结果。运行

```bash
nohup bash scripts/fig1/fig1_task1_scgen_hvg.sh > fig1_task1_scgen_hvg.log 2>&1
```

选定最佳高变基因数（default）进行 task1 的全部测评。运行

```bash
nohup bash scripts/fig1/fig1_task1_scgen.sh > fig1_task1_scgen.log 2>&1
```



```
nohup bash scripts/add_gaus/scgen.sh > add_gaus_scgen.log 2>&1
```



**5  DDPM**

测试不同高变基因梯度的评估结果

```bash
nohup bash scripts/fig1/fig1_task1_ddpm_hvg.sh > fig1_task1_ddpm_hvg.log 2>&1
```

选定最佳高变基因数（1000）进行 task1 的全部测评

```bash
nohup bash scripts/fig1/fig1_task1_ddpm.sh > fig1_task1_ddpm.log 2>&1
```





```
nohup bash scripts/add_gaus/ddpm.sh > add_gaus_ddpm.log 2>&1
```



**6  DDPM+MLP**

测试不同高变基因梯度的评估结果

```bash
nohup bash scripts/fig1/fig1_task1_ddpm_mlp_hvg.sh > fig1_task1_ddpm_mlp_hvg.log 2>&1
```

使用4000

```bash
nohup bash scripts/fig1/fig1_task1_ddpm_mlp.sh > fig1_task1_ddpm_mlp.log 2>&1
```



```
nohup bash scripts/add_gaus/ddpm_mlp.sh > add_gaus_ddpm_mlp.log 2>&1
```



#### Task 2

**0 Get the data**

```
python scripts/tools/fig1_task2.py
```

**1  Squidiff**

获取测评结果。运行

```bash
nohup bash scripts/fig1/fig1_task2_squidff.sh > fig1_task2_squidff.log 2>&1
```

**2  scDiff**

获取测评结果

```bash
nohup bash scripts/fig1/fig1_task2_scdiff.sh > fig1_task2_scdiff.log 2>&1
```

**3  scDiffusion**

```bash
nohup bash scripts/fig1/fig1_task2_scdiffusion.sh > fig1_task2_scdiffusion.log 2>&1
```

**4  scGen**

获取测评结果。运行

```bash
nohup bash scripts/fig1/fig1_task2_scgen.sh > fig1_task2_scgen.log 2>&1
```

**5  DDPM**

获取评测结果。运行

```bash
nohup bash scripts/fig1/fig1_task2_ddpm.sh > fig1_task2_ddpm.log 2>&1
```

**6  DDPM+MLP**

```bash
nohup bash scripts/fig1/fig1_task2_ddpm_mlp.sh > fig1_task2_ddpm_mlp.log 2>&1
```

#### Task 3

**0 Get the data**

```
# 获取原始数据集
python scripts/tools/fig1_task3.py
# 获取高变基因数据集
python scripts/tools/fig1_task3_hvg.py
```

**1  Squidiff**

依据 task1 中选取的最佳高变基因数（1000）进行测评

```bash
nohup bash scripts/fig1/fig1_task3_squidff.sh > fig1_task3_squidff.log 2>&1
```

**2  scDiff**

依据 task1 中选取的最佳高变基因数（default）进行测评

```bash
nohup bash scripts/fig1/fig1_task3_scdiff.sh > fig1_task3_scdiff.log 2>&1
```

**3  scDiffusion**

```bash
nohup bash scripts/fig1/fig1_task3_scdiffusion.sh > fig1_task3_scdiffusion.log 2>&1
```

**4  scGen**

```bash
nohup bash scripts/fig1/fig1_task3_scgen.sh > fig1_task3_scgen.log 2>&1
```

**5  DDPM**

依据 task1 中选取的最佳高变基因数（1000）进行测评

```bash
nohup bash scripts/fig1/fig1_task3_ddpm.sh > fig1_task3_ddpm.log 2>&1
```

**6  DDPM+MLP**

依据 task1 中选取的最佳高变基因数（4000）进行测评

```bash
nohup bash scripts/fig1/fig1_task3_ddpm_mlp.sh > fig1_task3_ddpm_mlp.log 2>&1
```

#### Task 4 

**0  Get the data**

1. 将 exp.csv 和 meta.csv 合并为 .h5ad 数据。运行

   ```bash
   bash scripts/tools/fig1_task4_merge.sh
   ```

   得到 `task4_ACTA2_control.h5ad`，`task4_ACTA2_coculture.h5ad`，`task4_ACTA2_IFN.h5ad`，`task4_B2M_control.h5ad`，`task4_B2M_coculture.h5ad`和`task4_B2M_IFN.h5ad`数据文件。

2. 划分方式 1：输入control预测coculture（训练集:测试集=8:2），输入control预测IFN（训练集:测试集=8:2）。运行

   ```bash
   bash scripts/tools/fig1_task4_split_1.sh
   ```

   得到`task4_B2M_control_coculture_train.h5ad`，`task4_B2M_control_coculture_test.h5ad`等共八个数据文件。注意，由于control和coculture（其他数据集也一样）的基因序列并不相同，直接合并会出现 nan 值，这里采用了取并集然后将 nan 变为 0 的通用做法。

3. 划分方式2：训练时control预测IFN，测试时control预测coculture。运行

   ```bash
   python scripts/tools/create_global_gene_list.py
   ```

   统一基因空间，基因数目为5737。然后运行

   ```bash
   bash scripts/tools/fig1_task4_split_2.sh
   ```

   得到`task4_ACTA2_control_to_coculture.h5ad`，`task4_ACTA2_control_to_ifn.h5ad`，`task4_B2M_control_to_coculture.h5ad`和`task4_B2M_control_to_ifn.h5ad`四个数据文件。

**1  Squidiff**

1. 在第一种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_1_squidiff.sh > fig1_task4_1_squidiff.log 2>&1
   ```

2. 在第二种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_2_squidiff.sh > fig1_task4_2_squidiff.log 2>&1
   ```

**2  scDiff**

1. 在第一种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_1_scdiff.sh > fig1_task4_1_scdiff.log 2>&1
   ```

2. 在第二种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_2_scdiff.sh > fig1_task4_2_scdiff.log 2>&1
   ```

**3  scDiffusion**

1. 在第一种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_1_scdiffusion.sh > fig1_task4_1_scdiffusion.log 2>&1
   ```

2. 在第二种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_2_scdiffusion.sh > fig1_task4_2_scdiffusion.log 2>&1
   ```

**4  scGen**

1. 在第一种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_1_scgen.sh > fig1_task4_1_scgen.log 2>&1
   ```

2. 在第二种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_2_scgen.sh > fig1_task4_2_scgen.log 2>&1
   ```

**5  DDPM**

1. 在第一种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_1_ddpm.sh > fig1_task4_1_ddpm.log 2>&1
   ```

2. 在第二种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_2_ddpm.sh > fig1_task4_2_ddpm.log 2>&1
   ```

**6  DDPM+MLP**

1. 在第一种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_1_ddpm_mlp.sh > fig1_task4_1_ddpm_mlp.log 2>&1
   ```

2. 在第二种划分方式下，获取测评结果，运行

   ```bash
   nohup bash scripts/fig1/fig1_task4_2_ddpm_mlp.sh > fig1_task4_2_ddpm_mlp.log 2>&1
   ```

### Fig 2

#### Task 1

**0  获取数据**

将 exp.csv 和 meta.csv 合并为 .h5ad 数据，并合并为训练集和测试集。运行

```bash
bash scripts/tools/fig2_task1_merge.sh
```

得到`seed123_control_test.h5ad`、`seed123_control_train.h5ad`等数据集。

**1  Squidiff**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task1_squidiff.sh > fig2_task1_squidiff.log 2>&1
```

**2  scDiff**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task1_scdiff.sh > fig2_task1_scdiff.log 2>&1
```

**3  scDiffusion**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task1_scdiffusion.sh > fig2_task1_scdiffusion.log 2>&1
```

**3  scGen**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task1_scgen.sh > fig2_task1_scgen.log 2>&1
```

**5  DDPM**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task1_ddpm.sh > fig2_task1_ddpm.log 2>&1
```

**6 DDPM+MLP**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task1_ddpm_mlp.sh > fig2_task1_ddpm_mlp.log 2>&1
```

#### Task 2

**1  Squidiff**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task2_squidiff.sh > fig2_task2_squidiff.log 2>&1
```

**2  scDiff**

受原代码限制，不进行该实验。

**3  scDiffusion**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task2_scdiffusion.sh > fig2_task2_scdiffusion.log 2>&1
```

**4  scGen**

受原代码限制，不进行该实验。

**5  DDPM**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task2_ddpm.sh > fig2_task2_ddpm.log 2>&1
```

**6 DDPM+MLP**

获取测评结果，运行

```bash
nohup bash scripts/fig2/fig2_task2_ddpm_mlp.sh > fig2_task2_ddpm_mlp.log 2>&1
```

#### Task 3

**0  Get the data**

将 exp.csv 和 meta.csv 合并为 .h5ad 数据。运行

```bash
bash scripts/tools/fig2_task3.sh
```

You will get `mouse_control_ifn.h5ad`等四个数据。

**1  Squidiff**

获取测评结果。运行

```bash
nohup bash scripts/fig2/fig2_task3_squidiff.sh > fig2_task3_squidiff.log 2>&1
```

**2  scDiff**

获取测评结果。运行

```bash
nohup bash scripts/fig2/fig2_task3_scdiff.sh > fig2_task3_scdiff.log 2>&1
```

**3  scDiffusion**

获取测评结果。运行

```bash
nohup bash scripts/fig2/fig2_task3_scdiffusion.sh > fig2_task3_scdiffusion.log 2>&1
```

**4  scGen**

获取测评结果。运行

```bash
nohup bash scripts/fig2/fig2_task3_scgen.sh > fig2_task3_scgen.log 2>&1
```

**5  DDPM**

获取测评结果。运行

```bash
nohup bash scripts/fig2/fig2_task3_ddpm.sh > fig2_task3_ddpm.log 2>&1
```

**2  DDPM+MLP**

获取测评结果。运行

```bash
nohup bash scripts/fig2/fig2_task3_ddpm_mlp.sh > fig2_task3_ddpm_mlp.log 2>&1
```



```
我有json数据形如```[{
        "conversations": [
            {
                "from": "human",
                "value": "Who was the father of the father of psychoanalysis?"
            },
            {
                "from": "gpt",
                "value": "<think>thought content xxx ... ...</think>normal content xxx ... ...<tool_call>\n{\"name\": \"tool_1\", \"arguments\": {\"query\": \"argument content 1\"}}\n</tool_call><tool_call>\n{\"name\": \"tool_2\", \"arguments\": {\"query\": \"argument content 2\"}}\n</tool_call>"
            },
            {
                "from": "human",
                "value": "<tool_response>response content ... ... </tool_response>"
            },
            {
                "from": "gpt",
                "value": "same ... ..."
            },
            {
                "from": "human",
                "value": "<tool_response>same ... ...</tool_response>"
            },
            {
                "from": "gpt",
                "value": "same ... ... <answer>final answer</answer>"
            }
        ],
        "tools": "[{\"type\": \"function\", \"function\": {\"name\": \"execute_code\", \"description\": \"Execute Python code in the specified conda environment\", \"parameters\": {\"type\": \"object\", \"properties\": {\"code\": {\"type\": \"string\", \"description\": \"Python code to execute\"}, \"filename\": {\"type\": \"string\", \"description\": \"Optional: Name of the file to save the code (default: generated UUID)\"}}, \"required\": [\"code\"]}}}, ... ...]",
        "system": "... ..."
    },``` 我需要你把这种数据拆分转换为两种数据，一种形如```{
  "_id": {
    "$oid": "689810fde3df02e840971b23"
  },
  "_class_id": "Record.MCPRecord",
  "final_answer": "Amir-Abbas Hoveyda",
  "right_answer": "Morarji Desai",
  "score": null,
  "split": "train",
  "status": "completed",
  "task": {
    "$ref": "Task",
    "$id": {
      "$oid": "689810d40e8073b07770979c"
    }
  },
  "trained_count": 0,
  "traj": [
    {
      "$ref": "DispatchedSamplingTask",
      "$id": {
        "$oid": "689810fde3df02e840971b24"
      }
    },
    {
      "$ref": "DispatchedSamplingTask",
      "$id": {
        "$oid": "6898110ee3df02e840971b25"
      }
    }
  ],
  "traj_id": 0
}```，即轨迹大纲。另一种如DispatchedSamplingTask 689810fde3df02e840971b24对应的到目前assistant + tool的具体trace形如```{
  "_id": {
    "$oid": "689810fde3df02e840971b24"
  },
  "_class_id": "DispatchedSamplingTask",
  "creat_time": {
    "$date": "2025-08-10T03:24:45.489Z"
  },
  "finish_time": {
    "$date": "2025-08-10T03:25:02.753Z"
  },
  "is_minio_managed": false,
  "priority": 0,
  "req_type": "chatcompletions",
  "request": {
    "messages": [
      {
        "role": "system",
        "content": "You are ... ..."
      },
      {
        "role": "user",
        "content": "Your task is to ... ... "
      }
      {"role": "assistant", ......}
      {"role": "tool", ......}
    ],
    "model": "train-model",
    "tools": [ # all the tools
      {
        "type": "function",
        "function": {
          "name": "execute_code",
          "description": "Execute Python code ......",
          "parameters": {
            "type": "object",
            "properties": {
              "code": {
                "type": "string",
                "description": "Python code to execute"
              },
              "filename": {
                "type": "string",
                "description": "Optional: Name of the file to save the code (default: generated UUID)"
              }
            },
            "required": [
              "code"
            ]
          }
        }
      },
      ... ...
    ]
  },
  "response": {
    "id": "c7637349bebc42249e0d653cf8bf890e",
    "choices": [
      {
        "finish_reason": "tool_calls",
        "index": 0,
        "logprobs": {
          "content": [
            {
              "token": "<think>",
              "bytes": [
                60,
                116,
                104,
                105,
                110,
                107,
                62
              ],
              "logprob": 0,
              "top_logprobs": []
            },
            ... ...
          ],
          "refusal": null
        },
        "message": {
          "content": "",
          "refusal": null,
          "role": "assistant",
          "annotations": null,
          "audio": null,
          "function_call": null,
          "tool_calls": [
            {
              "id": "call_f6d96a4e00614091ba626c40",
              "function": {
                "arguments": "{\"plan_steps\": [\"1. Identify the first place mentioned by name in the Book of Esther (NIV). [completed]\", \"2. Determine the Prime Minister of that place in April 1977. [completed]\"], \"next_step_goal\": \"Provide the final answer\", \"chosen_servers\": []}",
                "name": "manage_context"
              },
              "type": "function",
              "index": null
            }
          ],
          "reasoning_content": "Okay, <think> content </think>"
        },
        "matched_stop": null
      }
    ],
    "created": 1754796302,
    "model": "train-model",
    "object": "chat.completion",
    "service_tier": null,
    "system_fingerprint": null,
    "usage": {
      "completion_tokens": 603,
      "prompt_tokens": 7703,
      "total_tokens": 8306,
      "completion_tokens_details": null,
      "prompt_tokens_details": null
    }
  },
  "sampled_from": {
    "$ref": "InferenceService",
    "$id": {
      "$oid": "689810c3e3df02e840971b20"
    }
  },
  "score": null,
  "status": "completed",
  "task": {
    "$ref": "Task",
    "$id": {
      "$oid": "689810d40e8073b07770979c"
    }
  },
  "traj_id": 0,
  "type": "task"
}```
```

