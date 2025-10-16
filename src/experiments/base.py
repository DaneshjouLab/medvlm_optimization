# This source file is part of the Daneshjou Lab project
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
import dspy


class BaseExperiment(ABC):
    @abstractmethod
    def run(self) -> Tuple[List[dspy.Example], List[dspy.Example], dspy.Module, Callable]:
        pass