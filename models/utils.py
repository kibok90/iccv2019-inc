import math
import torch
import torch.nn as nn

class MultiHeadLinear(nn.ModuleList):
    def __init__(self, in_features, out_features, bias=True, no_grad=True):
        super(MultiHeadLinear, self).__init__()

        self.out_features = out_features
        for _ in range(out_features):
            self.append(nn.Linear(in_features, 1, bias))

        # parameters for future tasks should not be trainable
        if no_grad:
            for param in self.parameters():
                param.requires_grad = False

    def set_trainable(self, ts=[]):
        if not isinstance(ts, (list, range)):
            ts = [ts]
        for t, m in enumerate(self):
            requires_grad = (t in ts)
            for param in m.parameters():
                param.requires_grad = requires_grad

    def forward(self, x):
        y = torch.cat([self[t](x) for t in range(self.out_features)], dim=1)
        return y

# Kaiming initialization
def init_module(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            nn.init.normal_(m.weight.data, mean=0., std=math.sqrt(2. / fan_in))
            if m.bias is not None: nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight.data)
            if m.bias is not None: nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.Linear):
            fan_in = m.in_features
            nn.init.normal_(m.weight.data, mean=0., std=math.sqrt(2. / fan_in))
            if m.bias is not None: nn.init.zeros_(m.bias.data)
