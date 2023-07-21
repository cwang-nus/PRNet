import torch.nn as nn
import torch
from src.base.model import BaseModel

class PRNet(BaseModel):
    def __init__(self,
                 n_layers=6,
                 n_filters=64,
                 t_params=(12, 3, 3),
                 s_flag=True,
                 c_flag=True,
                 ext_flag=False,
                 x_flag=True,
                 s_r=8,
                 **args):
        super(PRNet, self).__init__(**args)
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.out_channels = self.output_dim*self.pred_step
        self.c_flag, self.ext_flag, self.x_flag = c_flag, ext_flag, x_flag
        self.c, self.p, self.t = t_params

        self.emb = nn.Sequential(nn.Conv2d(self.input_dim * self.c, n_filters, 1))

        self.prednet = SCEEncoder(n_filters, s_r, n_layers, s_flag=s_flag)

        if c_flag:
            if self.x_flag:
                self.fusenet = nn.Sequential(nn.Conv2d(3*n_filters, n_filters, kernel_size=1, bias=False))
            else:
                self.fusenet = nn.Sequential(nn.Conv2d(2*n_filters, n_filters, kernel_size=1, bias=False))

        self.out_conv = nn.Sequential(nn.BatchNorm2d(n_filters), nn.Conv2d(n_filters, self.out_channels, 1, 1, 0))

    def forward(self, inputs, ext=None):

        # get weekly segments x
        xt = inputs[2][:, -self.t:, -self.pred_step:] #b*t*timestep*D*H*W
        b, t, ts, c, h, w = xt.size()

        # get weekly segments y
        yt = inputs[-1]   # b*t*ts*in_dim*h*w
        yt = yt.view(b*t, ts, -1, h, w)

        # get recent segments
        if self.c_flag:
            xc = inputs[0][:, -self.prev_step:]  # b*ts*in_dim*H*W
            xt = xt.view(b*t, ts, -1, h, w)

            # concat recent segments and weekly segments to batch
            out = torch.cat([xc, xt, yt], dim=0)
        else:
            out = yt

        out = out.reshape(out.size(0), -1, h, w)
        out = self.emb(out)

        # SCE Encoder
        out = self.prednet(out)

        # diff and fuse function
        if self.c_flag:
            bs = t * b
            xc = out[:b]
            xc = xc.view(b, 1, -1, h, w)
            xt = out[b:b+bs]
            xt = xt.view(b, t, -1, h, w)
            yt = out[-bs:]
            yt = yt.view(b, t, -1, h, w)
            diff_x = xc - xt                        # b*t*filter*H*W
            out = torch.cat([diff_x, yt], dim=2)    # b*t*filter*H*W

            if self.x_flag:
                xc = xc.repeat(1, t, 1, 1, 1)
                out = torch.cat([xc, out], dim=2)

            out = out.view(b*t, -1, h, w)
            out = self.fusenet(out)

        out = self.out_conv(out)
        out = out.reshape(b, t, self.pred_step, -1, h, w)

        return out

class SCEEncoder(nn.Module):
    def __init__(self, n_filters, s_r, n_layers, s_flag=True):
        super().__init__()

        sce_blocks = []
        for _ in range(n_layers):
            sce_blocks.append(SCEBlock(n_filters, s_r, s_flag=s_flag))
        self.net = nn.Sequential(*sce_blocks)

        self.pred_conv = nn.Conv2d(n_filters, n_filters, 1)

    def forward(self, out):
        out1 = out                   # bs*n_filters*H*W
        out = self.net(out1)         # bs*n_filters*H*W --> b*n_filters*H*W
        out2 = self.pred_conv(out)   # bs*n_filters*H*W
        out = torch.add(out1, out2)  # bs*n_filters*H*W

        return out

class SCEBlock(nn.Module):
    def __init__(self, in_features, s_r, s_flag=True):
        super(SCEBlock, self).__init__()

        # standard cnn module
        conv_block = [nn.BatchNorm2d(in_features),
                      nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features)
                      ]

        self.conv_block = nn.Sequential(*conv_block)

        self.se = SCELayer(in_features, s_r, s_flag=s_flag)


    def forward(self, x):
        out = self.conv_block(x)
        out = self.se(out)
        out = out * torch.tanh(out)
        return x + out               # bs*f_emb*H*W

class SCELayer(nn.Module):
    def __init__(self, channel, s_r=8, s_flag=True, c_reduction=4):
        super(SCELayer, self).__init__()

        self.s_flag = s_flag

        if s_flag:
            self.s_pooling = s_r
            self.max_pool = nn.AdaptiveMaxPool2d(s_r)

            self.spatial = nn.Sequential(
                nn.Linear(s_r*s_r, 8, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(8, s_r*s_r, bias=False),
                nn.Sigmoid())

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // c_reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // c_reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        out = x

        # spatial enhanced module
        if self.s_flag:
            s = self.max_pool(x).view(b, c, -1)     # bs*f_emb*(s_r*s_r)
            s = self.spatial(s)
            x = s.view(b, c, self.s_pooling, self.s_pooling)  # bs*f*(s_r*s_r)

        # channel enhanced module
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)     # bs*channel
        return out * y.expand_as(out)       # bs*channel*H*W
