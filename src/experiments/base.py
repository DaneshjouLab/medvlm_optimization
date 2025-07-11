from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
import dspy


class BaseExperiment(ABC):
    @abstractmethod
    def run(self) -> Tuple[List[dspy.Example], List[dspy.Example], dspy.Module, Callable]:
        pass 