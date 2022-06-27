from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from aiaccel.optimizer.nelder_mead.sampler import NelderMeadSampler
from typing import Optional
import copy
import numpy as np


class NelderMeadSearchOptimizer(AbstractOptimizer):
    """An optimizer class with nelder mead algorithm.

    Attributes:
        nelder_mead ():
        parameter_pool ():
    """

    def __init__(self, options: dict) -> None:
        """Initial method of NelderMeadSearchOptimizer.

        Args:
            config (str): A file name of a configuration.
        """
        super().__init__(options)
        self.sampler = NelderMeadSampler(self.config)
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
            number (Optional[int]):
                A number of generating parameters.

        Returns:
            None

        Raises:
            TypeError: Causes when an invalid parameter type is set.
        """
        self.sampler.search()

        for _ in range(number):
            if len(self.sampler.get_parameter_pool()) == 0:
                self.logger.info('All parameters in pool has been generated.')
                break

            params, pool_p = self.sampler.generate_parameter()
            # Note: params
            # [
            #   {'parameter_name': param.name, 'type': param.type, 'value': value},
            #   {'parameter_name': param.name, 'type': param.type, 'value': value},
            #   {'parameter_name': param.name, 'type': param.type, 'value': value}
            #   ...
            # ]
            name = self.create_parameter_file({'parameters': params})
            self.sampler.update_ready_parameter_name(pool_p, name)
            self.sampler.add_order({'name': name, 'parameters': params})

    def _serialize(self) -> dict:
        """Serialize this module.

        Returns:
            dict: The serialized objects.
        """
        parameter_pool = copy.deepcopy(self.sampler.parameter_pool)
        for p_pool in parameter_pool:
            for p_pool_param in p_pool['parameters']:
                if type(p_pool_param['value']) is np.float64:
                    p_pool_param['value'] = float(p_pool_param['value'])

        self.serialize_datas = {
            'loop_count': self.loop_count,
            'parameter_pool': parameter_pool,
            'nelder_mead': self.sampler.nelder_mead.serialize(),
            'generated_parameter': self.sampler.generated_parameter,
            'order': self.sampler.order
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
        parameter_pool = copy.deepcopy(dict_objects['parameter_pool'])
        print(type(parameter_pool))
        for p_pool in parameter_pool:
            for p_pool_param in p_pool['parameters']:
                if type(p_pool_param['value']) is float:
                    p_pool_param['value'] = np.float64(p_pool_param['value'])

        self.sampler.initialize()
        self.sampler.set_parameter_pool(parameter_pool)
        self.sampler.nelder_mead.deserialize(dict_objects['nelder_mead'])
        self.sampler.order = dict_objects['order']
        self.sampler.generated_parameter = dict_objects['generated_parameter']
