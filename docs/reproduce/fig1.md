# Figure 1 Reproduction

### Highly variable gene gradient

In the data of Task 1 in Figure 1, the CD4T cell type has the largest number of cells (5,564) and is therefore chosen as the representative.

**Preprocessing (in order):**

1. Run `python preprocess_data/fig1/fig1_task1.py` to convert the `.csv` files in `data_ori/fig1/raw_task1` into `.h5ad` format.
2. Run `python preprocess_data/get_the_hvg_data_for_fig1.py` to extract highly variable gene (HVG) data from the resulting `.h5ad` files.

**Evaluation:** Run the following commands to obtain results for each baseline:

```
nohup bash scripts/highly_variable_gene_gradient/ddpm_hvg.sh > ddpm_hvg.log 2>&1 &
nohup bash scripts/highly_variable_gene_gradient/ddpm_mlp_hvg.sh > ddpm_mlp_hvg.log 2>&1 &
nohup bash scripts/highly_variable_gene_gradient/scdiff_hvg.sh > scdiff_hvg.log 2>&1 &
nohup bash scripts/highly_variable_gene_gradient/scgen_hvg.sh > scgen_hvg.log 2>&1 &
nohup bash scripts/highly_variable_gene_gradient/squidiff_hvg.sh > squidiff_hvg.log 2>&1 &
nohup bash scripts/highly_variable_gene_gradient/scdiffusion_hvg.sh > scdiffusion_hvg.log 2>&1 &
```

Each script runs three trials and writes the per-run and averaged metrics to the log, and also outputs a CSV file for easy tabulation.

**Note:** You may need to set your own data and model paths (e.g. via the `ROOT_DIR` and `ANNOTATION_MODEL_DIR` environment variables or the defaults at the top of each script).

### Fig 1

#### Task 1

**Get the data**

Since, overall, the models trained on the data with the lowest number of highly variable genes (1000) achieved the best performance, the experiments of Task 1 and Task 3 in Figure 1 are conducted using the processed data with 1000 HVGs extracted from the original data.  

**Run the evaluation**

```bash
nohup bash scripts/fig1/fig1_task1/fig1_task1_ddpm_mlp.sh > fig1_task1_ddpm_mlp.log 2>&1 &
nohup bash scripts/fig1/fig1_task1/fig1_task1_ddpm.sh > fig1_task1_ddpm.log 2>&1 &
nohup bash scripts/fig1/fig1_task1/fig1_task1_scgen.sh > fig1_task1_scgen.log 2>&1 &
nohup bash scripts/fig1/fig1_task1/fig1_task1_scdiff.sh > fig1_task1_scdiff.log 2>&1 &
nohup bash scripts/fig1/fig1_task1/fig1_task1_scdiffusion.sh > fig1_task1_scdiffusion.log 2>&1 &
nohup bash scripts/fig1/fig1_task1/fig1_task1_squidff.sh > fig1_task1_squidff.log 2>&1 &
```

#### Task 2

**Get the data**

```bash
python preprocess_data/fig1/fig1_task2.py
```

**Run the evaluation**

```bash
nohup bash scripts/fig1/fig1_task2_ddpm_mlp.sh > fig1_task2_ddpm_mlp.log 2>&1 &
nohup bash scripts/fig1/fig1_task2_ddpm.sh > fig1_task2_ddpm.log 2>&1 &
nohup bash scripts/fig1/fig1_task2_scgen.sh > fig1_task2_scgen.log 2>&1 &
nohup bash scripts/fig1/fig1_task2_scdiff.sh > fig1_task2_scdiff.log 2>&1 &
nohup bash scripts/fig1/fig1_task2_scdiffusion.sh > fig1_task2_scdiffusion.log 2>&1 &
nohup bash scripts/fig1/fig1_task2_squidff.sh > fig1_task2_squidff.log 2>&1 &
```

#### Task 3

**Get the data**

```bash
python preprocess_data/fig1/fig1_task3.py
```

**Run the evaluation**

```
nohup bash scripts/fig1/fig1_task3_ddpm_mlp.sh > fig1_task3_ddpm_mlp.log 2>&1 &
nohup bash scripts/fig1/fig1_task3_ddpm.sh > fig1_task3_ddpm.log 2>&1 &
nohup bash scripts/fig1/fig1_task3_scgen.sh > fig1_task3_scgen.log 2>&1 &
nohup bash scripts/fig1/fig1_task3_scdiff.sh > fig1_task3_scdiff.log 2>&1 &
nohup bash scripts/fig1/fig1_task3_scdiffusion.sh > fig1_task3_scdiffusion.log 2>&1 &
nohup bash scripts/fig1/fig1_task3_squidff.sh > fig1_task3_squidff.log 2>&1 &
```

#### Task 4 

**Get the data**

1. **Merge `exp.csv` and `meta.csv` into `.h5ad` format**

   Run the following script:

   ```bash
   bash preprocess_data/fig1/fig1_task4_merge.sh
   ```

   This will generate the following `.h5ad` files: `task4_ACTA2_control.h5ad`, `task4_ACTA2_coculture.h5ad`, `task4_ACTA2_IFN.h5ad`, `task4_B2M_control.h5ad`, `task4_B2M_coculture.h5ad`, `task4_B2M_IFN.h5ad`.

2. **Split Strategy 1**

   * Train on *control* and predict *coculture* (train:test = 8:2)
   * Train on *control* and predict *IFN* (train:test = 8:2)

   Run:

   ```bash
   bash preprocess_data/fig1/fig1_task4_split_1.sh
   ```

   This will generate eight `.h5ad` files, including: `task4_B2M_control_coculture_train.h5ad`, `task4_B2M_control_coculture_test.h5ad` (and corresponding files for other gene groups).

   **Note:**
   Since the gene sets in *control* and *coculture* (and similarly in other dataset pairs) are not identical, directly merging them would introduce `NaN` values. To address this, we take the union of genes and replace missing values (`NaN`) with zeros as a standard preprocessing step.

3. **Split Strategy 2**

   * Training: control → IFN
   * Testing: control → coculture

   First, unify the gene space:

   ```bash
   python preprocess_data/fig1/create_global_gene_list.py
   ```

   This produces a unified gene list containing **5,737 genes**.

   Then run:

   ```bash
   bash preprocess_data/fig1/fig1_task4_split_2.sh
   ```

   This will generate four `.h5ad` files: `task4_ACTA2_control_to_coculture.h5ad`, `task4_ACTA2_control_to_ifn.h5ad`, `task4_B2M_control_to_coculture.h5ad`, `task4_B2M_control_to_ifn.h5ad`.

**Run the evaluation**
```
nohup bash scripts/fig1/fig1_task4_1_squidiff.sh > fig1_task4_1_squidiff.log 2>&1 &
nohup bash scripts/fig1/fig1_task4_2_squidiff.sh > fig1_task4_2_squidiff.log 2>&1 &

nohup bash scripts/fig1/fig1_task4_1_scdiff.sh > fig1_task4_1_scdiff.log 2>&1 &
nohup bash scripts/fig1/fig1_task4_2_scdiff.sh > fig1_task4_2_scdiff.log 2>&1 &

nohup bash scripts/fig1/fig1_task4_1_scdiffusion.sh > fig1_task4_1_scdiffusion.log 2>&1 &
nohup bash scripts/fig1/fig1_task4_2_scdiffusion.sh > fig1_task4_2_scdiffusion.log 2>&1 &

nohup bash scripts/fig1/fig1_task4_1_scgen.sh > fig1_task4_1_scgen.log 2>&1 &
nohup bash scripts/fig1/fig1_task4_2_scgen.sh > fig1_task4_2_scgen.log 2>&1 &

nohup bash scripts/fig1/fig1_task4_1_ddpm.sh > fig1_task4_1_ddpm.log 2>&1 &
nohup bash scripts/fig1/fig1_task4_2_ddpm.sh > fig1_task4_2_ddpm.log 2>&1 &

nohup bash scripts/fig1/fig1_task4_1_ddpm_mlp.sh > fig1_task4_1_ddpm_mlp.log 2>&1 &
nohup bash scripts/fig1/fig1_task4_2_ddpm_mlp.sh > fig1_task4_2_ddpm_mlp.log 2>&1 &
```
