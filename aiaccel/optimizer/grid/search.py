from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from aiaccel.optimizer.grid.sampler import GridSampler
from typing import Optional


class GridSearchOptimizer(AbstractOptimizer):
    """An optimizer class with grid search algorithm.

    Attributes:
        ready_params (List[dict]): A list of ready hyper parameters.
        generate_index (int): A number of generated hyper parameters.
    """

    def __init__(self, options: dict) -> None:
        """Initial method of GridSearchOptimizer.

        Args:
            config (str): A file name of a configuration.
        """
        super().__init__(options)
        self.sampler = GridSampler(self.config)
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
        params = []
        for _ in range(number):
            param = self.sampler.generate_parameter()
            if param is None:
                self.logger.info('Generated all of parameters.')
                self.all_parameter_generated = True
                return
            params.append(param)
            self.generated_parameter += 1
        self.create_parameter_files(params)

    def _serialize(self) -> None:
        """Serialize this module.

        Returns:
            dict: The serialized objects.
        """
        self.serialize_datas = {
            'loop_count': self.loop_count,
            'sampler': self.sampler
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
