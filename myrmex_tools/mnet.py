import torch
import numpy as np
import torch.nn as nn

from torch import Tensor
from collections import OrderedDict

def _conv2D_outshape(in_shape, Cout, kernel, padding=(0,0), stride=(1,1), dilation=(1,1)):
    if len(in_shape)==2:
        Hin, Win = in_shape
    else:
        _, Hin, Win = in_shape

    Hout = np.int8(np.floor(((Hin+2*padding[0]-dilation[0]*(kernel[0]-1)-1)/(stride[0]))+1))
    Wout = np.int8(np.floor(((Win+2*padding[1]-dilation[1]*(kernel[1]-1)-1)/(stride[1]))+1))
    return (Cout, Hout, Wout)

def to_tensor(x): 
    if not isinstance(x, Tensor) and x is not None: 
        if isinstance(x, np.ndarray): return torch.from_numpy(x)
        return torch.Tensor(x)
    else: return x

class MyrmexNet(nn.Module):

    def __init__(self, 
        input_dim = [16,16],
        kernel_sizes = [(5,5), (3,3)],
        cnn_out_channels = [32, 64],
        conv_stride = (2,2),
        conv_padding = (0,0),
        nchannels = 1, # 1 if single / merged samples, 2 if both sensor samples simultaneously
        nclasses = 5,
        fc_neurons = [128, 64],
        ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.kernel_sizes = kernel_sizes
        self.cnn_out_channels = cnn_out_channels
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.nchannels = nchannels
        self.nclasses = nclasses
        self.fc_neurons = fc_neurons

        self.conv_outshape = None # will be set by self._make_conv
        self.cnn_in_channels = [self.nchannels] + self.cnn_out_channels[:-1]
        self.conv = self._make_conv()

        self.mlp = nn.Sequential(
            nn.Linear(np.prod(self.conv_outshape), self.fc_neurons[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.fc_neurons[0], self.fc_neurons[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.fc_neurons[1], self.nclasses),
        )

    def forward(self, x: Tensor):
        """
        x.shape = [batch, C, height, width]

        C can be ...
            1 if we use single readings or merged samples
            2 if we use both sensor readings simultaneously
        
        our samples are square, so height = width
        height = width = 16 if we don't cut outer edges
        """
        x = to_tensor(x)

        # simple forward pass, first conv layers, then mlp
        convout = self.conv(x)
        mlpout = self.mlp(convout)

        return mlpout

    def _make_conv(self):
        """
        construct convnet that takes a single myrmex sample, passes it through N conv units and flattens it. 
        sets `self.conv_outshape` to be the flatten layer's number of neurons
        """

        layers = []
        for i, (kern, inc, outc) in enumerate(
            zip(
                self.kernel_sizes,
                self.cnn_in_channels,
                self.cnn_out_channels
            )):

            # construct Conv2D unit: conv + ReLu
            layers.append((f"conv2d_{i}", nn.Conv2d(
                    in_channels=inc,
                    out_channels=outc,
                    kernel_size=kern,
                    stride=self.conv_stride,
                    padding=self.conv_padding)
            ))
            # layers.append((f"batch_norm_{i}_{name}", nn.BatchNorm2d(outc, momentum=0.01))) # worse performance with BN in empirical trials
            layers.append((f"relu_conv_{i}", nn.ReLU(inplace=True)))# why inplace?

            self.conv_outshape = _conv2D_outshape(
                self.input_dim if self.conv_outshape is None else self.conv_outshape,
                Cout=outc,
                padding=self.conv_padding,
                kernel=kern,
                stride=self.conv_stride
            )

        layers.append((f"flatten_conv", nn.Flatten()))
        return nn.Sequential(OrderedDict(layers))