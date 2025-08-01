"""Definition of the abstract base for any flucs system.

Outlines the basic functionality of any system using
abstract methods.

"""

from abc import ABC, abstractmethod
from importlib.resources import files
from flucs import FlucsInput

class FlucsSystem(ABC):

    input : FlucsInput = None

    @classmethod
    def load_defaults(cls, flucs_input : FlucsInput):
        resource_path = files(cls.__module__) / "defaults.toml"
        with resource_path.open("r") as f:
            contents = f.read()

        flucs_input.load_toml_str(contents, default=True)


    @abstractmethod
    def initialise(self) -> None:
        pass


    def __init__(self, flucs_input : FlucsInput) -> None:
        self.input = flucs_input
