from pertdiffbench.backends.chemcpa import ChemCPABackend
from pertdiffbench.backends.ddpm import DDPBackend
from pertdiffbench.backends.ddpm_mlp import DDPMMLPBackend
from pertdiffbench.backends.encoder import EncoderBackend
from pertdiffbench.backends.scgen import ScGenBackend
from pertdiffbench.backends.scdiff import ScDiffBackend
from pertdiffbench.backends.scdiffusion import ScDiffusionBackend
from pertdiffbench.backends.squidiff import SquidiffBackend

MODEL_REGISTRY = {
    "ddpm": DDPBackend(),
    "ddpm_mlp": DDPMMLPBackend(),
    "squidiff": SquidiffBackend(),
    "scdiffusion": ScDiffusionBackend(),
    "scgen": ScGenBackend(),
    "scdiff": ScDiffBackend(),
    "chemcpa": ChemCPABackend(),
    "encoder": EncoderBackend(),
}

SUPPORTED_MODELS = list(MODEL_REGISTRY.keys())

__all__ = ["MODEL_REGISTRY", "SUPPORTED_MODELS"]
