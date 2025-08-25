"""
pipeline.config.py - Simple dataclass for pipeline params.
"""
# pylint: disable=[C0116, C0321]

from dataclasses import dataclass, field

from simulator.config import SimulatorConfig
from translator.config import TranslatorConfig

@dataclass
class PipelineConfig:
    """
    Parent config object for pipeline orchestration.
    """
    fault_tolerant: bool = False
    verbose: bool = False
    save_dir: str = "results/pipeline"
    sim_config: SimulatorConfig = field(default_factory=SimulatorConfig)
    translator_config: TranslatorConfig = field(default_factory=TranslatorConfig)

    def with_ft_only(self, ft: bool = True): self.fault_tolerant = ft; return self
    def with_verbose(self, verbose: bool = True): self.verbose = verbose; return self
    def with_save_dir(self, d: str): self.save_dir = d; return self
    def with_sim_config(self, sim_config: SimulatorConfig):
        self.sim_config = sim_config
        return self
    def with_translator_config(self, translator_config: TranslatorConfig):
        self.translator_config = translator_config
        return self
