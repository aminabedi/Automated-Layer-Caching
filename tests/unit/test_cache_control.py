import pytest
import torch
import numpy as np
from unittest import mock
from backbone.modules import CacheControl, Config

@pytest.fixture()
def mock_config(mock_configuration):
    with mock.patch('backbone.modules.Config', spec=Config) as MockConfig:
        mock_config = mock.create_autospec(
            Config, instance=True
        )
        mock_config.reader= mock_configuration
        mock_config.__dict__.update(mock_configuration.__dict__)
        def update_config(entries):
            mock_config.__dict__.update(entries)
        
        mock_config.update.side_effect = update_config
        MockConfig.return_value = mock_config
        yield MockConfig

@pytest.fixture()
def mock_cache_model_all_miss(mock_config):
    def return_all_zero(inp):
        print(inp.shape)
        return torch.from_numpy(np.zeros((inp.shape[0], mock_config().reader["num_classes"].get(int))))
    with mock.patch('backbone.NConvMDense.NConvMDense', autospec=True, instance=True, spec_set=True, side_effect = return_all_zero) as mock_cache_model:
        yield mock_cache_model

@pytest.fixture()
def mock_cache_model_all_hit(mock_config):
    with mock.patch('backbone.NConvMDense.NConvMDense', autospec=True, instance=True, spec_set=True) as mock_cache_model:
        mock_cache_model.__call__ = mock.Mock()
        mock_cache_model.__call__.side_effect = lambda inp: torch.from_numpy(np.zeros(inp.shape[0], mock_config.num_classes))
        yield mock_cache_model


def test_cc_configs_set_correctly(mock_config, mock_cache_model_all_miss):
    mock_config=mock_config()
    exits = mock_config.reader["exits"].get(dict)

    cc = CacheControl(exits=exits, cache_models= [mock_cache_model_all_miss] * len(exits.keys()))

    assert cc.conf.device == mock_config.reader["inference"]["device"].get(str)
    assert cc.conf.num_classes == mock_config.reader["num_classes"].get(int)
    assert cc.conf.training == mock_config.reader["training"]["is_training"].get(bool)
    assert cc.conf.shrink == mock_config.reader["inference"]["shrink"].get(bool)


def test_cc_avoids_uncached_layers(mock_config, mock_cache_model_all_miss, mock_input):
    mock_config=mock_config()
    exits = mock_config.reader["exits"].get(dict)
    cc = CacheControl(exits=exits, cache_models= [mock_cache_model_all_miss] * len(exits.keys()))
    
    ret = cc.exit(mock_input, layer_id=-10)

    assert np.array_equal(ret, mock_input)

def test_cc_cache_misses(mock_config, mock_cache_model_all_miss, mock_input):
    mock_config=mock_config()
    exits = mock_config.reader["exits"].get(dict)
    cc = CacheControl(exits=exits, cache_models= [mock_cache_model_all_miss] * len(exits.keys()))
    cc.batch_received(mock_input)

    ret = cc.exit(mock_input, layer_id=list(exits.keys())[0])

    assert np.array_equal(ret, mock_input)