# models/geneformer_ddpm_mlp_diffusion.py

import torch
from torch import nn
import scanpy as sc
# NEW: Import Hugging Face transformers library
from transformers import AutoTokenizer, AutoModelForMaskedLM
from geneformer import TranscriptomeTokenizer

from .base import DiffusionModel
# REMOVED: scGPT utils are no longer needed
# from utils.scgpt_utils import load_scgpt, embed_cells
from .mlp_ddpm_mlp_autoencoder import ScRNADecoder
from .gaussian_diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
# Note: The file 'mlp_ddpm_mlp_diffusion.py' was referenced in the original code,
# which might be a typo and should point to a file containing SinusoidalPosEmb.
# Assuming it's available in the same directory or a utils path.
from .mlp_ddpm_mlp_diffusion import SinusoidalPosEmb


class TimeConditionalWrapper(nn.Module):
    """Adds time-step embedding to the core_net, making its signature forward(z, t)."""
    def __init__(self, core_net: nn.Module, time_dim: int, latent_dim: int):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(time_dim)
        self.fc_t = nn.Linear(time_dim, latent_dim)
        self.net = core_net

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 1) Construct time embedding
        te = self.time_emb(t)      # [B, time_dim]
        te = self.fc_t(te)         # [B, latent_dim]
        # 2) Add time embedding to z and pass to the core_net
        return self.net(z + te)


# NEW: Helper function to embed cells using GeneFormer
@torch.no_grad()
def embed_cells_with_geneformer(
    model: nn.Module,
    tokenizer,
    adata_batch: torch.Tensor,
    gene_ids: list,
    device: torch.device
) -> torch.Tensor:
    """
    Embeds a batch of cells using a pre-trained GeneFormer model.
    """
    model.eval()
    
    # Convert the sparse batch to a dense numpy array if necessary
    if not isinstance(adata_batch, torch.Tensor):
        if hasattr(adata_batch, "toarray"):
            adata_batch = adata_batch.toarray()
        else:
            adata_batch = adata_batch.numpy()
    else:
        adata_batch = adata_batch.cpu().numpy()

    # Create gene lists for each cell
    cell_gene_sequences = []
    for i in range(adata_batch.shape[0]):
        # Get indices of non-zero genes for the current cell
        gene_indices = adata_batch[i].nonzero()[0]
        # Map indices to gene names/IDs
        cell_genes = [gene_ids[j] for j in gene_indices]
        # Join into a space-separated string for tokenization
        cell_gene_sequences.append(" ".join(cell_genes))

    # Tokenize the sequences
    inputs = tokenizer(cell_gene_sequences, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    
    # Get embeddings from GeneFormer
    outputs = model(**inputs)
    
    # Use mean pooling of the last hidden state to get a fixed-size embedding for each cell
    # (B, SeqLen, EmbDim) -> (B, EmbDim)
    embedding = outputs.last_hidden_state.mean(dim=1)
    
    return embedding


# MODIFIED: Renamed class from MLPDDPMMLPscGPT to MLPDDPMMLPGeneFormer
class MLPDDPMMLPGeneFormer(DiffusionModel):
    def __init__(self, cfg):
        T     = cfg.model.diffusion.T
        betas = torch.linspace(cfg.model.diffusion.beta_1,
                               cfg.model.diffusion.beta_T,
                               T)
        super().__init__(T, betas)

        device = torch.device(cfg.train.device)

        # 1) MODIFIED: Load GeneFormer model and tokenizer from Hugging Face
        model_name = cfg.model.geneformer.model_name
        self.geneformer_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        tk = TranscriptomeTokenizer({"cell_type": "cell_type", "organ_major": "organ"}, nproc=4)
        tk.tokenize_data("你的输入数据目录/", "你的输出目录/", "输出文件前缀")
        
        # REMOVED: The hook registration for scGPT is no longer applicable.

        adata_ref = sc.read_h5ad(cfg.data.path)
        self.gene_ids = adata_ref.var_names.to_list()

        # 2) Build core_net: reconstructs the latent_dim
        #    This now correctly uses the latent_dim from the config (e.g., 256 for GeneFormer)
        latent_dim = cfg.model.decoder.latent_dim
        hidden_dim = cfg.model.decoder.hidden_dim
        core_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # 3) Wrap it with a time-conditional wrapper
        time_dim = cfg.model.diffusion.hidden_dim
        conditioned_net = TimeConditionalWrapper(core_net, time_dim, latent_dim)

        # 4) Trainer & Sampler (unconditional process also calls conditioned_net(z, t))
        self.trainer = GaussianDiffusionTrainer(
            model=conditioned_net,
            beta_1=cfg.model.diffusion.beta_1,
            beta_T=cfg.model.diffusion.beta_T,
            T=T,
            conditional=False
        )
        self.sampler = GaussianDiffusionSampler(
            model=conditioned_net,
            beta_1=cfg.model.diffusion.beta_1,
            beta_T=cfg.model.diffusion.beta_T,
            T=T
        )

        # 5) Decoder
        self.decoder = ScRNADecoder(
            latent_dim,
            cfg.model.decoder.output_dim,
            hidden_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) MODIFIED: Use GeneFormer to get z0 embedding
        z0 = embed_cells_with_geneformer(
            self.geneformer_model, self.geneformer_tokenizer,
            x, self.gene_ids, self.betas.device
        ).float()  # [B, latent_dim]
        
        # 2) Construct zero-filled time-steps
        B = z0.shape[0]
        t = torch.zeros(B, dtype=torch.long, device=z0.device)
        
        # 3) Pass to the trainer
        return self.trainer(z0, t)

    @torch.no_grad()
    def sample(self, adata_ref):
        # 1) MODIFIED: Get control embedding using GeneFormer
        z0 = embed_cells_with_geneformer(
            self.geneformer_model, self.geneformer_tokenizer,
            adata_ref.X, self.gene_ids, self.betas.device
        ).float()
        
        # 2) Initialize noise
        z_t = torch.randn_like(z0)
        B = z0.shape[0]
        
        # 3) Iterative sampling
        for step in reversed(range(self.T)):
            t = torch.full((B,), step, dtype=torch.long, device=z0.device)
            eps = self.trainer.model(z_t, t)
            mean = self.sampler.predict_xt_prev_mean_from_eps(z_t, t, eps)
            var = self.sampler.posterior_var[step]
            if step > 0:
                z_t = mean + torch.sqrt(var) * torch.randn_like(z_t)
            else:
                z_t = mean
        
        # 4) Decode
        return self.decoder(z_t).clamp(-1, 1)