from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from aiaccel.util.filesystem import get_file_hp_finished
from aiaccel.util.filesystem import interprocess_lock_file
from aiaccel.optimizer.sobol.sampler import SobolSampler
from sobol_seq import i4_sobol
from typing import Optional
import aiaccel
import fasteners


class SobolSearchOptimizer(AbstractOptimizer):
    """An optimizer class with sobol algorithm.

    Attributes:
        generate_index (int): A number of generated hyper parameters.

    ToDo: The development of original library was stopped. It's recommended to
        be replaced with SciPy sobol module.
    """

    def __init__(self, options: dict) -> None:
        """Initial method of SobolSearchOptimizer.

        Args:
            config (str): A file name of a configuration.
        """
        super().__init__(options)
        self.sampler = SobolSampler(self.config)

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

        params = []

        initial_parameter = self.generate_initial_parameter()
        if initial_parameter is not None:
            params.append(initial_parameter)
            number -= 1

        for _ in range(number):
            param = self.sampler.generate_parameter()
            params.append(param)

        self.create_parameter_files(params)

    def _serialize(self) -> dict:
        """Serialize this module.

        Returns:
            dict: The serialized objects.
        """
        self.serialize_datas = {
            'sampler': self.sampler,
            'generated_parameter': self.sampler.generated_parameter,
            'generate_index': self.sampler.generate_index,
            'loop_count': self.loop_count
        }
        return super()._serialize()

    def _deserialize(self, dict_objects: dict) -> None:
        """Deserialize this module.

        Args:
            dict_objects(dict): A dictionary including serialized objects.

        Returns:
            None
        """
        super()._deserialize(dict_objects)
        self.sampler = dict_objects['sampler']
        self.sampler.generated_parameter = dict_objects['generated_parameter']
        self.sampler.generate_index = dict_objects['generate_index']
