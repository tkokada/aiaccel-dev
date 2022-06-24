from pathlib import Path
import aiaccel
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from aiaccel.util.filesystem import get_file_hp_finished
from aiaccel.util.filesystem import interprocess_lock_file
from aiaccel.parameter import load_parameter
from aiaccel.config import Config
from sobol_seq import i4_sobol
from typing import Optional
import fasteners


class SobolSampler:
    def __init__(self, config: Config):
        self.config = config
        self.logger = None
        self.ws = Path(self.config.workspace.get()).resolve()
        self.params = load_parameter(self.config.hyperparameters.get())
        self.dict_hp = self.ws / aiaccel.dict_hp
        self.dict_lock = self.ws / aiaccel.dict_lock
        self.generated_parameter = 0
        self.generate_index = None

    def set_logger(self, logger):
        self.logger = logger

    def initialize(self):
        with fasteners.InterProcessLock(
            interprocess_lock_file(self.dict_hp, self.dict_lock)
        ):
            files = get_file_hp_finished(self.ws)
        self.generate_index = len(files)

    def generate_parameter(self) -> dict:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.

        Returns:
            None
        """
        l_params = self.params.get_parameter_list()
        n_params = len(l_params)
        vec, seed = i4_sobol(n_params, self.generate_index)
        self.generate_index = seed

        params = []
        for i in range(0, n_params):
            min_value = l_params[i].lower
            max_value = l_params[i].upper
            value = (max_value - min_value) * vec[i] + min_value
            params.append(
                {
                    'parameter_name': l_params[i].name,
                    'type': l_params[i].type,
                    'value': float(value)
                }
            )

        self.generated_parameter += 1
        return {'parameters': params}
