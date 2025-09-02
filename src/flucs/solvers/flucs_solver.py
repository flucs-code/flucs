"""Definition of the abstract base for any flucs solver.

Outlines the basic functionality of any solver using
abstract methods.

"""

import enum
from abc import ABC, abstractmethod
from importlib.resources import files
from flucs.input import FlucsInput
from flucs.systems import FlucsSystem


class FlucsSolverState(enum.Enum):
    NOTINITIALISED = enum.auto()
    INITIALISED = enum.auto()
    TIMING = enum.auto()
    RUNNING = enum.auto()
    PAUSED = enum.auto()
    DONE = enum.auto()


class FlucsSolver[T_System: FlucsSystem](ABC):
    input : FlucsInput
    system : T_System
    state : FlucsSolverState

    @classmethod
    def load_defaults(cls, flucs_input : FlucsInput):
        resource_path = files(cls.__module__) / "defaults.toml"
        with resource_path.open("r") as f:
            contents = f.read()

        flucs_input.load_toml_str(contents, default=True)

    @abstractmethod
    def run(self) -> None:
        pass

    def __init__(self, flucs_input : FlucsInput, flucs_system : T_System) -> None:
        self.input = flucs_input
        self.system = flucs_system
        self.state = FlucsSolverState.NOTINITIALISED
