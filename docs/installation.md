# Installation

### 🛠️ Configure the environment

```
conda create -n pertdiffbench python=3.10 -y && conda activate pertdiffbench
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 
pip install omegaconf numpy anndata tqdm scanpy gdown einops torch_geometric adjustText wandb zarr blobfile
pip install git+https://github.com/LouiseDck/scgen
pip install -e .
sudo apt update
sudo apt install openmpi-bin libopenmpi-dev
pip install mpi4py
```
