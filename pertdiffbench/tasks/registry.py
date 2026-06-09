from pertdiffbench.tasks.cross_celltype import (
    CrossCelltypeExtendTask,
    CrossCelltypePlusTask,
    CrossCelltypeTask,
)
from pertdiffbench.tasks.cross_species import CrossSpeciesLooTask, CrossSpeciesTask
from pertdiffbench.tasks.encoder import EncoderTask
from pertdiffbench.tasks.known_condition import KnownConditionTask
from pertdiffbench.tasks.moa import MoaDiffTask, MoaSameTask
from pertdiffbench.tasks.noise import NoiseTask
from pertdiffbench.tasks.temporal import TemporalTask

TASK_REGISTRY = {
    "known_condition": KnownConditionTask(),
    "cross_celltype": CrossCelltypeTask(),
    "cross_celltype_extend": CrossCelltypeExtendTask(),
    "cross_celltype_plus": CrossCelltypePlusTask(),
    "cross_species": CrossSpeciesTask(),
    "cross_species_loo": CrossSpeciesLooTask(),
    "moa_same": MoaSameTask(),
    "moa_diff": MoaDiffTask(),
    "temporal": TemporalTask(),
    "noise": NoiseTask(),
    "encoder": EncoderTask(),
}

SUPPORTED_TASKS = list(TASK_REGISTRY.keys())

__all__ = ["TASK_REGISTRY", "SUPPORTED_TASKS"]
