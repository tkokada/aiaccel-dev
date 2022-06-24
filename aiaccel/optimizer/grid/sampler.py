import math
from pathlib import Path
from typing import List, Tuple, Union, Optional
from functools import reduce
from operator import mul
import aiaccel
from aiaccel.config import Config
from aiaccel.parameter import HyperParameter
from aiaccel.parameter import load_parameter
from aiaccel.util.filesystem import get_file_hp_finished
from aiaccel.util.filesystem import get_file_hp_ready
from aiaccel.util.filesystem import get_file_hp_running


class GridSampler:
    def __init__(self, config: Config):
        self.config = config
        self.logger = None
        self.ws = Path(self.config.workspace.get()).resolve()
        self.params = load_parameter(self.config.hyperparameters.get())
        self.dict_lock = self.ws / aiaccel.dict_lock
        self.ready_params = None
        self.generate_index = None

    def initialize(self):
        self.ready_params = []
        for param in self.params.get_parameter_list():
            self.ready_params.append(self.generate_grid_points(param))
        self.generate_index = self.get_num_of_generated_indexes()

    def set_logger(self, logger):
        self.logger = logger

    def check_format(self):
        pass

    def get_grid_options(
        self,
        parameter_name: str,
    ) -> Tuple[Union[int, None], bool, Union[int, None]]:

        """Get options about grid search.

        Args:
            parameter_name (str)   : A parameter name to get its options.
            config (ConfileWrapper): A config object.

        Returns:
            Tuple[Union[int, None], bool, Union[int, None]]: The first one is a
                base of logarithm parameter. The second one is logarithm parameter
                or not. The third one is a step of the grid.

        Raises:
            KeyError: Causes when step is not specified.
        """
        base = None
        logscale = False
        step = None

        grid_options = self.config.hyperparameters.get()

        for g in grid_options:
            if g['name'] == parameter_name:
                if 'step' in g.keys():
                    step = float(g['step'])
                else:
                    step = None
                logscale = bool(g['log'])
                if logscale:
                    base = int(g['base'])
                break

        if step is None:
            raise KeyError(f'No grid option for parameter: {parameter_name}')
        else:
            return base, logscale, step

    def generate_grid_points(self, param: HyperParameter) -> dict:
        """Make a list of all parameters for this grid.

        Args:
            param (HyperParameter): A hyper parameter object.
            config (ConfileWrapper): A configuration object.

        Returns:
            dict: A dictionary including all grid parameters.

        Raises:
            TypeError: Causes when an invalid parameter type is set.
        """
        new_param = {
            'parameter_name': param.name,
            'type': param.type
        }

        if param.type.lower() == 'int' or param.type.lower() == 'float':
            base, logscale, step = self.get_grid_options(param.name)
            lower = param.lower
            upper = param.upper
            n = int((upper - lower) / step) + 1

            if logscale is True:
                lower_x = lower ** base
                upper_x = upper ** base
                x = lower_x
                new_param['parameters'] = []
                while x < upper_x or math.isclose(x, upper_x, abs_tol=1e-10):
                    new_param['parameters'].append(math.log(x, base))
                    x += step
                new_param['parameters'].append(upper)
            else:
                if param.type.lower() == 'int':
                    new_param['parameters'] = [int(lower + i * step) for i in range(0, n)]
                elif param.type.lower() == 'float':
                    new_param['parameters'] = [lower + i * step for i in range(0, n)]
                else:
                    assert False

        elif param.type.lower() == 'categirical':
            new_param['parameters'] = list(param.choices)

        elif param.type.lower() == 'ordinal':
            new_param['parameters'] = list(param.sequence)

        else:
            raise TypeError('Invalid parameter type: {}'.format(param.type))

        return new_param

    def get_num_of_generated_indexes(self):
        ready_files = get_file_hp_ready(self.ws, self.dict_lock)
        running_files = get_file_hp_running(self.ws, self.dict_lock)
        finished_files = get_file_hp_finished(self.ws, self.dict_lock)

        return (
            len(ready_files) +
            len(running_files) +
            len(finished_files)
        )

    def get_next_parameter_index(self) -> Union[List[int], None]:
        """Get a next parameter index.

        Returns:
            Union[List[int], None]: It returns None if all parameters are
                already generated.
        """
        parameter_lengths = [len(i['parameters']) for i in self.ready_params]
        remain = self.generate_index
        max_index = reduce(mul, parameter_lengths)

        if self.generate_index >= max_index:
            self.logger.warning('All parameters were generated.')
            return None

        parameter_index = []
        div = [
            reduce(
                lambda x, y: x * y,
                parameter_lengths[0:-1 - i]
            ) for i in range(0, len(parameter_lengths) - 1)
        ]

        for i in range(0, len(parameter_lengths) - 1):
            d = int(remain / div[i])
            parameter_index.append(d)
            remain -= d * div[i]

        parameter_index.append(remain)
        self.generate_index += 1

        return parameter_index

    def generate_parameter(self, number: Optional[int] = 1) -> dict:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.

        Returns:
            None
        """

        parameter_index = self.get_next_parameter_index()
        new_params = []

        if parameter_index is None:
            return

        for i in range(0, len(self.ready_params)):
            new_param = {
                'parameter_name': self.ready_params[i]['parameter_name'],
                'type': self.ready_params[i]['type'],
                'value': self.ready_params[i]['parameters'][parameter_index[i]]
            }
            new_params.append(new_param)
        return {'parameters': new_params}
