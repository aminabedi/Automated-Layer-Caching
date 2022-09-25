

from torch.nn import LazyLinear, Module, ReLU, LogSoftmax, Softmax, LazyConv2d, Sequential
import torch

class NConvMDense(Module):
    def __init__(self, exit_config):
        super(NConvMDense, self).__init__()
        c = exit_config["convs"][0]
        # self.t = [Conv2d(c["in_ch"], c["out_ch"], c["k"], stride = c["s"]) for c in exit_config["convs"]
        self.convs = Sequential(*[LazyConv2d(c["out_ch"], c["k"], stride = c["s"]) for c in exit_config["convs"]])
        self.linears = Sequential(*[Sequential(LazyLinear(l["out"]), ReLU()) for l in exit_config["linears"]])
        self.LogSoftmax = LogSoftmax(dim=1)
    
    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), -1)
        out = self.linears(out)
        out = self.LogSoftmax(out)
        return out

