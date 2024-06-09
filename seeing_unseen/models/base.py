import abc

import torch.nn as nn


class SPModel(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abc.abstractmethod
    def forward(self, **kwargs):
        pass
