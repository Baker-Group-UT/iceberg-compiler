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
    verbose: bool = True
    save: bool = False
    save_dir: str = "results/pipeline"
    sim_config: SimulatorConfig = field(default_factory=SimulatorConfig)
    translator_config: TranslatorConfig = field(default_factory=TranslatorConfig)
    seed: int = None

    def __post_init__(self):
        """
        Push `fault_tolerant` down to the translator, so the constructor kwarg works.

        Only `with_ft` used to reach `translator_config.fault_tolerant`. The constructor
        field was read by nothing but the DataFrame writer, so
        `PipelineConfig(fault_tolerant=True)` translated in bare mode and then labelled
        every row `fault_tolerant=True` -- 47 gates where `with_ft()` gives 110, filed as
        FT. Same shape as the `ft_chance` misspelling below.

        Guarded so that passing an explicit `translator_config` with FT already on is not
        switched back off by the default `False` here.
        """
        if self.fault_tolerant:
            self.translator_config.fault_tolerant = True
        else:
            self.fault_tolerant = bool(self.translator_config.fault_tolerant)

        # `seed` now reaches the simulator too, so one knob reproduces a whole scan.
        # It used to seed permutation *selection* only, while the shot stream came from
        # `sim_config.seed` and was never threaded -- so two runs at the same
        # `config.seed` drew identical permutations and different fidelities, and a
        # "seed sweep" confounded sampling noise with placement choice. Only filled in
        # when `sim_config` has no seed of its own, so an explicit one still wins.
        if self.seed is not None and self.sim_config.seed is None:
            self.sim_config.seed = self.seed

    def with_ft(self, ft: bool = True, ft_chance: float = -1.0):
        """
        Opt in to fault-tolerant gate encodings. Bare mode is the default.

        Note that `ft_chance` currently goes nowhere: `TranslatorConfig` declares
        `fault_tolerance_chance` and nothing reads it. This line used to set
        `fault_tolerant_chance` -- a different spelling -- which on a plain dataclass
        silently created a *new* attribute and left the declared field at -1.0, so
        `with_ft(True, ft_chance=0.5)` did nothing twice over. The names agree now, which
        means wiring the knob up is a one-place change rather than a debugging session.
        """
        self.fault_tolerant = ft
        self.translator_config.fault_tolerant = ft
        self.translator_config.fault_tolerance_chance = ft_chance
        return self
    def with_verbose(self, verbose: bool = True): self.verbose = verbose; return self
    def with_save_dir(self, d: str): self.save_dir = d; return self
    def with_sim_config(self, sim_config: SimulatorConfig):
        self.sim_config = sim_config
        return self
    def with_translator_config(self, translator_config: TranslatorConfig):
        self.translator_config = translator_config
        return self
    def with_seed(self, seed: int): self.seed = seed; return self
