import pytest
import numpy as np
from unittest import mock
from backbone.modules import CacheControl, Config

@pytest.fixture()
def mock_config():
    with mock.patch('backbone.modules.Config', autospec=True, spec_set=True):
        mock_config = Config(name="TestConfig", config_file='tests/config_test.yaml')
        mock_config.update({
            "device": "cpu", 
            "num_classes": 10,
            "training": False,
            "shrink": True
        })
        yield mock_config

@pytest.fixture()
def mock_cache_model_all_miss(mock_config):
    with mock.patch('backbone.NConvMDense.NConvMDense', autospec=True, instance=True, spec_set=True) as mock_cache_model:
        mock_cache_model.__call__ = mock.Mock()
        mock_cache_model.__call__.side_effect =  lambda inp: np.zeros(inp.shape[0], mock_config.num_classes)
        yield mock_config

def test_no_exit_with_threshold_equal_one(mock_config, mock_cache_model_all_miss):
    config = mock_config()
    print(mock_config.__dict__)
    cc = CacheControl(exits= [config.reader["exits"].get(dict).keys()], cache_models= [mock_cache_model_all_miss for i in range(len(config.reader["exits"].get(dict).keys()))])
    assert False