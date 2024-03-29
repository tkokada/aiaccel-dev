from aiaccel.optimizer.sobol.search import SobolOptimizer
from tests.base_test import BaseTest
import pytest


class TestSobolOptimizer(BaseTest):

    @pytest.fixture(autouse=True)
    def setup_optimizer(self, clean_work_dir):
        options = {
            'config': self.config_sobol_path,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'optimizer'
        }
        self.optimizer = SobolOptimizer(options)
        self.optimizer.storage.alive.init_alive()
        yield
        self.optimizer = None

    def test_pre_process(self):
        self.optimizer.storage.alive.init_alive()
        assert self.optimizer.pre_process() is None

    def test_generate_parameter(self):
        self.optimizer.pre_process()
        assert self.optimizer.generate_parameter() is None
