# Noise-Perturbed Data Experiments

## Noise-perturbed data

### Gaussian noise

From the repository root:

```bash
conda activate pertdiffbench && export PYTHONPATH=./
cd preprocess_data/noise_perturbation_exp
python cd4t_gaus.py
```

Perturbed data are written under `data/add_gaussian_noise_output`. **Run the script twice** if you need separate **train** and **validation** splits (as produced by your preprocessing configuration).

Then return to the repo root and launch the baselines:

```bash
cd ../../..
nohup bash scripts/noise_exp/gaussian_perturbed_data/ddpm_mlp.sh > gausnoise_ddpm_mlp.log 2>&1 &
nohup bash scripts/noise_exp/gaussian_perturbed_data/ddpm.sh > gausnoise_ddpm.log 2>&1 &
nohup bash scripts/noise_exp/gaussian_perturbed_data/scdiff.sh > gausnoise_scdiff.log 2>&1 &
nohup bash scripts/noise_exp/gaussian_perturbed_data/scdiffusion.sh > gausnoise_scdiffusion.log 2>&1 &
nohup bash scripts/noise_exp/gaussian_perturbed_data/scgen.sh > gausnoise_scgen.log 2>&1 &
nohup bash scripts/noise_exp/gaussian_perturbed_data/squidiff.sh > gausnoise_squidiff.log 2>&1 &
```

### Biological noise (log-normal)

```bash
conda activate pertdiffbench && export PYTHONPATH=./
cd preprocess_data/noise_perturbation_exp
python cd4t_log_norm.py
```

Outputs go to `data/add_lognormal_bionoise_output`. **Run twice** if you need both train and validation data.

```bash
cd ../../..
nohup bash scripts/noise_exp/lognormal_bionoise_perturbed_data/ddpm_mlp.sh > lognormal_ddpm_mlp.log 2>&1 &
nohup bash scripts/noise_exp/lognormal_bionoise_perturbed_data/ddpm.sh > lognormal_ddpm.log 2>&1 &
nohup bash scripts/noise_exp/lognormal_bionoise_perturbed_data/scdiff.sh > lognormal_scdiff.log 2>&1 &
nohup bash scripts/noise_exp/lognormal_bionoise_perturbed_data/scdiffusion.sh > lognormal_scdiffusion.log 2>&1 &
nohup bash scripts/noise_exp/lognormal_bionoise_perturbed_data/scgen.sh > lognormal_scgen.log 2>&1 &
nohup bash scripts/noise_exp/lognormal_bionoise_perturbed_data/squidiff.sh > lognormal_squidiff.log 2>&1 &
```

### Technical noise

#### Poisson

```bash
conda activate pertdiffbench && export PYTHONPATH=./
cd preprocess_data/noise_perturbation_exp
python cd4t_poisson.py
```

Outputs go to `data/add_poisson_technoise_output`. **Run twice** if you need both train and validation data.

```bash
cd ../../..
nohup bash scripts/noise_exp/poisson_technoise_perturbed_data/ddpm_mlp.sh > poisson_ddpm_mlp.log 2>&1 &
nohup bash scripts/noise_exp/poisson_technoise_perturbed_data/ddpm.sh > poisson_ddpm.log 2>&1 &
nohup bash scripts/noise_exp/poisson_technoise_perturbed_data/scdiff.sh > poisson_scdiff.log 2>&1 &
nohup bash scripts/noise_exp/poisson_technoise_perturbed_data/scdiffusion.sh > poisson_scdiffusion.log 2>&1 &
nohup bash scripts/noise_exp/poisson_technoise_perturbed_data/scgen.sh > poisson_scgen.log 2>&1 &
nohup bash scripts/noise_exp/poisson_technoise_perturbed_data/squidiff.sh > poisson_squidiff.log 2>&1 &
```

#### Zero inflation

```bash
conda activate pertdiffbench && export PYTHONPATH=./
cd preprocess_data/noise_perturbation_exp
python cd4t_zero_inflation.py
```

Outputs go to `data/add_zero_inflation_output`. **Run twice** if you need both train and validation data.

```bash
cd ../../..
nohup bash scripts/noise_exp/zero_inflation_technoise_perturbed_data/ddpm_mlp.sh > zero_inflation_ddpm_mlp.log 2>&1 &
nohup bash scripts/noise_exp/zero_inflation_technoise_perturbed_data/ddpm.sh > zero_inflation_ddpm.log 2>&1 &
nohup bash scripts/noise_exp/zero_inflation_technoise_perturbed_data/scdiff.sh > zero_inflation_scdiff.log 2>&1 &
nohup bash scripts/noise_exp/zero_inflation_technoise_perturbed_data/scdiffusion.sh > zero_inflation_scdiffusion.log 2>&1 &
nohup bash scripts/noise_exp/zero_inflation_technoise_perturbed_data/scgen.sh > zero_inflation_scgen.log 2>&1 &
nohup bash scripts/noise_exp/zero_inflation_technoise_perturbed_data/squidiff.sh > zero_inflation_squidiff.log 2>&1 &
```
