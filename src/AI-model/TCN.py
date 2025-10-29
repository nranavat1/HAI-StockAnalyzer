import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation, drop):
        super().__init__()
        pad = (k - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, k,
                               padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k,
                               padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(drop)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.bn2(out)
        out = F.gelu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return F.gelu(out + res)

class TCN(nn.Module):
    def __init__(self, in_ch, widths=(32, 32, 32), k=3, drop=0.05):
        super().__init__()
        layers = []
        for i in range(len(widths)):
            dilation = 2 ** i
            in_c = in_ch if i == 0 else widths[i-1]
            out_c = widths[i]
            layers.append(TemporalBlock(in_c, out_c, k, dilation, drop))
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(widths[-1], 1)

    def forward(self, x):
        # x: (B, C, T)
        y = self.network(x)
        # Use the last time step’s features
        y = y[:, :, -1]
        y = self.head(y)
        return y.squeeze(-1)
