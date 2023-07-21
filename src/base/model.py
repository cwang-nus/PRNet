from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __init__(self,
                 name: str,
                 dataset: str,
                 device,
                 height: int,
                 width: int,
                 prev_step: int,
                 pred_step: int,
                 input_dim: int,
                 output_dim: int):
        super(BaseModel, self).__init__()
        self.name = name
        self.dataset = dataset
        self.device = device
        self.height = height
        self.width = width
        self.prev_step = prev_step
        self.pred_step = pred_step
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)