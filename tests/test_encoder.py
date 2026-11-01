"""
For testing the Encoder.
"""

import torch, torch.nn as nn
from blackvip.coordinator.encoders import FrozenEncoder

class Dummy(FrozenEncoder):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.bn = nn.BatchNorm2d(4)


# A) gradients off only, still in train mode -> BN stats still change
print('--- Case A ---')
d = Dummy()
for p in d.parameters():
    p.requires_grad_(False)
before = d.bn.running_mean.clone()
input_ = torch.randn(2, 3, 8, 8)
d.bn(d.conv(input_))
print(before, "-->", d.bn.running_mean)


# B). 
print('--- Case B ---')
d = Dummy()
d._freeze()
print("training flags:", d.training, d.bn.training)
before = d.bn.running_mean.clone()
input_ = torch.randn(2, 3, 8, 8)
d.bn(d.conv(input_))
print('After freezing the grad')
print(before, "-->", d.bn.running_mean)


# C) parent asks for train mode -> must stay in eval
print('--- Case C ---')
d = Dummy(); d._freeze()
ret = d.train()                       # simulates coordinator.train()
print("returns self:", ret is d)                            # True
print("still eval:", not any(m.training for m in d.modules()))  # True
