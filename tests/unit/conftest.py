import pytest
from unittest.mock import patch
import confuse
import numpy as np
import torch

@pytest.fixture()
def mock_configuration():
    reader = confuse.Configuration('TestConfig')
    reader.set_file("tests/unit/config_test.yaml")
    return reader

@pytest.fixture()
def mock_input():
    return torch.from_numpy(np.random.rand(10, 32, 32))