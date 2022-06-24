import math
from pathlib import Path
from typing import List, Tuple, Union, Optional
from functools import reduce
from operator import mul
from typing import Dict, List, Optional, Union
import aiaccel
from aiaccel.config import Config
from aiaccel.parameter import HyperParameter
from aiaccel.parameter import load_parameter
from aiaccel.util.filesystem import get_file_hp_finished
from aiaccel.util.filesystem import get_file_hp_ready
from aiaccel.util.filesystem import get_file_hp_running


class RamdomSampler:
    def __init__(self, config: Config):
        self.config = config
        self.logger = None
        self.ws = Path(self.config.workspace.get()).resolve()
        self.params = load_parameter(self.config.hyperparameters.get())
        self.dict_lock = self.ws / aiaccel.dict_lock
        self.generated_parameter = 0

    def set_logger(self, logger):
        self.logger = logger

    def initialize(self):
        pass

    def generate_parameter(self) -> dict:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.

        Returns:
            None
        """

        new_params = []
        sample = self.params.sample()

        for s in sample:
            new_params.append(
                {
                    'parameter_name': s['name'],
                    'type': s['type'],
                    'value': s['value']
                }
            )
        self.generated_parameter += 1
        return {'parameters': new_params}
