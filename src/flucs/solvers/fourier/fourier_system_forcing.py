from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flucs.solvers.fourier.fourier_system import FourierSystem


class FourierSystemForcing(ABC):
    """
    Base class for optional forcing methods used by FourierSystem solvers.
    """

    linear: bool
    explicit: bool

    def __init__(self, system: FourierSystem):
        self.system = system

    @abstractmethod
    def setup_cuda_definitions(self) -> None:
        pass