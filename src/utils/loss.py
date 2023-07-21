import torch.nn as nn

class MAELoss(nn.Module):
    def __init__(self, ):
        super(MAELoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, y_hat, y):
        '''
        :param y_hat:   N**
        :param y:       N**
        :return:        MAE
        '''
        return self.loss(y_hat, y)

class SmoothL1Loss(nn.Module):
    def __init__(self, ):
        super(SmoothL1Loss, self).__init__()
        self.loss = nn.SmoothL1Loss()

    def forward(self, y_hat, y):
        '''
        :param y_hat:   N**
        :param y:       N**
        :return:        Smooth L1 Loss
        '''
        return self.loss(y_hat, y)