import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv1D(nn.Module):
    """
    Conv1D with optional BN and ReLU.
    """
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, relu=False, bn=False, bias=False):
        super(BasicConv1D, self).__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm1d(out_c, eps=1e-5, momentum=0.01, affine=True) if bn else nn.Identity()
        self.relu = nn.ReLU() if relu else nn.Identity()

    def forward(self, x):
        """
            x: (B, C, N)
        """
        return self.relu(self.bn(self.conv(x)))

class BasicConv2D(nn.Module):
    """
    Conv2D with optional BN and ReLU.
    """
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, relu=False, bn=False, bias=False):
        super(BasicConv2D, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.01, affine=True) if bn else nn.Identity()
        self.relu = nn.ReLU() if relu else nn.Identity()

    def forward(self, x):
        """
            x: (B, C, H, W)
        """
        return self.relu(self.bn(self.conv(x)))

class CLA(nn.Module):
    """
    Cross Layer Attention module.
    Args:
        in_ch (int): Number of input channels in feats (last layer).
        k_size (int): Adaptive selection of kernel size.
    """
    def __init__(self, in_ch, k_size=3):
        super(CLA, self).__init__()
        # input convolution to change spatial size
        self.in_conv = BasicConv2D(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
        # pool convolution to aggregate spatial information
        self.pool_conv = BasicConv2D(in_ch, 1, kernel_size=1)
        # attention convolution to generate attention map
        self.attn_conv = BasicConv2D(3, 1, kernel_size=k_size, padding=(k_size - 1) // 2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, feats1, feats2):
        residual = feats2

        # [B, C, H, W] -> [B, C, h, w]
        feats1 = self.in_conv(feats1) 

        # [B, C, h, w] -> [B, 1, h, w]
        max_out = torch.max(feats1, dim=1, keepdim=True)[0] 
        avg_out = torch.mean(feats1, dim=1, keepdim=True)
        conv_out = self.pool_conv(feats1) 

        # [B, 3, h, w]
        pool_out = torch.cat([max_out, avg_out, conv_out], dim=1)

        attn_map = self.attn_conv(pool_out)
        attn_map = self.sigmoid(attn_map) 

        return feats2 * attn_map.expand_as(feats2) + residual


class ECA(nn.Module):
    """Constructs a ECA module.
    modified from https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = BasicConv1D(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
            x: (B, C)
        """
        y = x.unsqueeze(1)  # [B, 1, C]
        # Two different branches of ECA module
        y = self.conv(y).squeeze(1)  # [B, C]
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class NormedLinear(nn.Module):
    """
    A custom linear layer with input and output feature normalization.
    copied from https://github.com/kaidic/LDAM-DRW/blob/master/models/resnet_cifar.py#L37-L46, better for long-tailed datasets
    """

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        """
        Performs forward pass of the linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).

        """
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable input, hidden, and output dimensions.
    """
    def __init__(self, dim_in, dim_mlp_list, dim_out, use_norm=False, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i, dim_mlp in enumerate(dim_mlp_list):
            in_ch = dim_in if i == 0 else dim_mlp_list[i-1]
            self.layers.append(nn.Sequential(
                nn.Linear(in_ch, dim_mlp),
                nn.Dropout(dropout),
                nn.LeakyReLU()
            ))
        
        in_ch = dim_mlp_list[-1] if len(dim_mlp_list) > 0 else dim_in
        self.layers.append(NormedLinear(in_ch, dim_out) if use_norm else nn.Linear(in_ch, dim_out))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
