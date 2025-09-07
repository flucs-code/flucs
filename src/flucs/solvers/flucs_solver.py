"""Definition of the abstract base for any flucs solver.

Outlines the basic functionality of any solver using
abstract methods.

"""

import enum
from typing import TypeVar, Generic
from abc import ABC, abstractmethod
from flucs.input import FlucsInput
from flucs.systems import FlucsSystem


class FlucsSolverState(enum.Enum):
    NOTINITIALISED = enum.auto()
    INITIALISED = enum.auto()
    TIMING = enum.auto()
    RUNNING = enum.auto()
    PAUSED = enum.auto()
    DONE = enum.auto()


T_System = TypeVar("T_System", bound="FlucsSystem")
class FlucsSolver(Generic[T_System], ABC):
    input: FlucsInput
    system: T_System
    state: FlucsSolverState

    @classmethod
    def load_defaults(cls, flucs_input: FlucsInput):
        from pathlib import Path
        import importlib

        module = importlib.import_module(cls.__module__)
        resource_path = Path(module.__file__).with_name("defaults.toml")
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
