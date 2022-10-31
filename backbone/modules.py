import torch
import torch.nn as nn
import confuse
from NConvMDense import NConvMDense
from typing import List, Tuple
import logging



class Config:
    def __init__(self, name="UnnamedConfig", config_file="config.yaml"):
        self.reader = confuse.Configuration(name)
        self.reader.set_file(config_file)

    def update(self, entries):
        self.__dict__.update(entries)


class CachableModule(nn.Module):
    cached_layers: List[int] = None
    cache_exits: nn.ModuleList = None

    def __init__(self, *args, **kwargs):
        super(CachableModule, self).__init__(*args, **kwargs)
        self.exits = Config().reader["exits"].get(dict)
        
    def forward(self, x):
        self.ensure_cache_init()
        self.cache_control.batch_received(x)
        x = self._forward(x)
        self.cache_control.exit(x, final=True)
        return self.cache_control.results
    
    def ensure_cache_init(self):
        if not hasattr(self, 'cache_models'):
            self.cache_models = nn.ModuleList([NConvMDense(e["architecture"]) for e in self.exits.values()]) # .load_state_dict(e["weights_file"])
            self.cache_control = CacheControl(self.exits, self.cache_models)


def threshold_confidence(x, threshold):
    x_exp = torch.exp(x)
    mx, _ = torch.max(x_exp, dim=1)
    return torch.gt(mx, threshold), mx


class CacheControl():
    def __init__(self, exits: dict, cache_models: nn.ModuleList, logger=None):
        self.build_conf()
        self.exits = exits
        self.num_exits = len(exits) + 1
        self.logger = logger
        self.cache_models = cache_models

    def build_conf(self):
        self.conf = Config()
        d = {
            "device": self.conf.reader["inference"]["device"].get(str),
            "num_classes": self.conf.reader["num_classes"].get(int),
            "training": self.conf.reader["training"]["is_training"].get(bool),
            "shrink": self.conf.reader["inference"]["shrink"].get(bool)
        }
        self.conf.update(d)

    def batch_received(self, batch: torch.Tensor):
        self.ret = torch.ones((batch.shape[0], self.conf.num_classes)).to(
            self.conf.device) * -1
        self.hits = [torch.ones(0) for _ in range(self.num_exits)]
        self.idxs = [torch.arange(batch.shape[0]).to(self.conf.device)]
        self.outputs = [torch.ones(0) for _ in range(self.num_exits)]
        self.vectors = []
        self.item_exits = torch.ones(batch.shape[0]).to(self.conf.device) * -1

    def exit(self, out, layer_id=-1, final=False)-> Tuple[torch.Tensor, bool]:
        if layer_id not in self.exits.keys() and not final:
            return out
        if self.conf.training and layer_id in self.exits.keys():
            self.vectors.append(out)
            return
        self.exit_idx = -1 if final else sorted(self.exits.keys()).index(layer_id)
        current_idxs = self.idxs[-1]
        if final:
            self.outputs[self.exit_idx] = out
            if not self.conf.shrink:
                self.ret = out
            if out.size(0):
                self.ret[current_idxs] = out
                self.item_exits[current_idxs] = torch.ones(
                    out.size(0)).to(self.conf.device) * -1
                self.hits[self.exit_idx] = torch.ones(
                    out.size(0)).to(self.conf.device)
            return out
        cache_pred = self.cache_models[self.exit_idx](out)
        print(self.exits, "Exit_idx", self.exit_idx)
        hits, mx = threshold_confidence(cache_pred, self.exits[layer_id]["threshold"])
        if self.logger:
            self.logger.info(f"Max cache conf: {mx}")

        self.outputs[self.exit_idx] = cache_pred
        self.hits[self.exit_idx] = hits
        if self.conf.shrink and hits.sum().item():
            no_hits = torch.logical_not(hits)
            hit_idxs = current_idxs[hits]
            current_idxs = current_idxs[no_hits]
            out = out[no_hits]
            self.ret[hit_idxs] = cache_pred[hits]
            self.item_exits[hit_idxs] = torch.ones(
                hit_idxs.size(0)).to(self.conf.device) * self.exit_idx
        if len(current_idxs):
            self.idxs.append(current_idxs)
            self.exit_idx += 1
        return out

    @property
    def results(self):
        return self.ret

    def report(self):
        d = self.__dict__
        return {x: d[x] for x in d if x not in ['ret', 'conf', 'cache_models']}
