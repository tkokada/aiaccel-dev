from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from aiaccel.optimizer.random.sampler import RamdomSampler
from typing import Optional
import copy
import numpy as np


class RandomSearchOptimizer(AbstractOptimizer):
    """An optimizer class with a random algorithm.

    """

    def __init__(self, options: dict) -> None:
        """Initial method of GridSearchOptimizer.

        Args:
            config (str): A file name of a configuration.
        """
        super().__init__(options)
        self.sampler = RamdomSampler(self.config)
        self.sampler.set_logger(self.logger)

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        super().pre_process()
        self.sampler.initialize()

    def generate_parameter(self, number: Optional[int] = 1) -> None:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.

        Returns:
            None
        """

        returned_params = []
        self.get_dict_state()
        initial_parameter = self.generate_initial_parameter()

        if initial_parameter is not None:
            returned_params.append(initial_parameter)
            number -= 1

        params = []
        for _ in range(number):
            param = self.sampler.generate_parameter()
            params.append(param)

        self.create_parameter_files(params)

    def _serialize(self) -> dict:
        """Serialize this module.

        Returns:
            dict: serialize data.
        """
        self.serialize_datas = {
            'generated_parameter': self.sampler.generated_parameter,
            'loop_count': self.loop_count
        }
        return super()._serialize()

    def _deserialize(self, dict_objects: dict) -> None:
        """ Deserialize this module.

        Args:
            dict_objects(dict): A dictionary including serialized objects.

        Returns:
            None
        """
        super()._deserialize(dict_objects)
        self.sampler.generated_parameter = dict_objects['generated_parameter']
