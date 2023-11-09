import torch
import torch.nn as nn
import torch.nn.functional as F


class SpexEncoder(nn.Module):
    def __init__(self, short_size=20, middle_size=80, long_size=160, speech_encoder_out_channels=256):
        super().__init__()

        self.encoder_1d_short = nn.Conv1d(1,
                                          speech_encoder_out_channels,
                                          short_size,
                                          stride=short_size // 2,
                                          padding=0)

        self.encoder_1d_middle = nn.Sequential(nn.ConstantPad1d((0, middle_size - short_size), 0),
                                               nn.Conv1d(1,
                                                         speech_encoder_out_channels,
                                                         middle_size,
                                                         stride=short_size // 2,
                                                         padding=0))

        self.encoder_1d_long = nn.Sequential(nn.ConstantPad1d((0, long_size - short_size), 0),
                                             nn.Conv1d(1,
                                                       speech_encoder_out_channels,
                                                       long_size,
                                                       stride=short_size // 2,
                                                       padding=0))

    def forward(self, x):
        w1 = F.relu(self.encoder_1d_short(x))
        w2 = F.relu(self.encoder_1d_middle(x))
        w3 = F.relu(self.encoder_1d_long(x))
        return w1, w2, w3


class SpexDecoder(nn.Module):
    def __init__(self, short_size=20, middle_size=80, long_size=160, speech_encoder_out_channels=256):
        super().__init__()

        self.decoder_1d_short = nn.ConvTranspose1d(speech_encoder_out_channels,
                                                   1,
                                                   short_size,
                                                   stride=short_size // 2)

        self.decoder_1d_middle = nn.ConvTranspose1d(speech_encoder_out_channels,
                                                    1,
                                                    middle_size,
                                                    stride=short_size // 2)

        self.decoder_1d_long = nn.ConvTranspose1d(speech_encoder_out_channels,
                                                  1,
                                                  long_size,
                                                  stride=short_size // 2)

    def forward(self, short, mid, long):
        dec_short = self.decoder_1d_short(short)
        dec_mid = self.decoder_1d_middle(mid)
        dec_long = self.decoder_1d_long(long)
        return dec_short, dec_mid, dec_long


class LayerNorm(nn.Module):
    def __init__(self, emb_channels):
        super().__init__()
        self.LN = nn.LayerNorm(emb_channels)

    def forward(self, x):
        return self.LN(x.transpose(1, 2)).transpose(1, 2)


class GlobalLNSpex(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(GlobalLNSpex, self).__init__()
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(dim, 1))
        self.gamma = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x):
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
        x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        return x


class TCNSpex(nn.Module):
    def __init__(self, in_channels=256, conv_channels=512, sp_channels=256, kernel_size=3, dilation=1):
        super().__init__()

        self.pad = dilation * (kernel_size - 1) // 2
        self.one_block = nn.Sequential(nn.Conv1d(in_channels + sp_channels, conv_channels, 1),
                                       nn.PReLU(),
                                       GlobalLNSpex(conv_channels),
                                       nn.Conv1d(conv_channels, conv_channels, kernel_size,
                                                 groups=conv_channels,
                                                 padding=self.pad,
                                                 dilation=dilation),
                                       nn.PReLU(),
                                       GlobalLNSpex(conv_channels),
                                       nn.Conv1d(conv_channels, in_channels, 1))

    def forward(self, x, sp_emb):
        if sp_emb is None:
            return self.one_block(x) + x
        else:
            proj = x.shape[-1]
            sp_ = torch.unsqueeze(sp_emb, -1)
            sp_ = sp_.repeat(1, 1, proj)
            out = torch.concat([x, sp_], 1)
            return self.one_block(out) + x


class TCNBlocksSpex(nn.Module):
    def __init__(self, in_channels=256, conv_channels=512, sp_channels=256, kernel_size=3, num_blocks=8):
        super().__init__()

        self.first = TCNSpex(in_channels=in_channels, conv_channels=conv_channels, sp_channels=sp_channels,
                             kernel_size=kernel_size, dilation=1)

        stack = [TCNSpex(in_channels=in_channels, conv_channels=conv_channels, sp_channels=0,
                         kernel_size=kernel_size, dilation=2 ** i) for i in range(1, num_blocks)]

        self.stack = nn.ModuleList(stack)

    def forward(self, x, sp_emb):
        x = self.first(x, sp_emb)
        for i in self.stack:
            x = i(x)
        return x


class ResNetSpex(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()

        self.conv_first = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        self.conv_second = nn.Conv1d(out_dims, out_dims, kernel_size=1, bias=False)
        self.batch_norm_first = nn.BatchNorm1d(out_dims)
        self.batch_norm_second = nn.BatchNorm1d(out_dims)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.maxpooling = nn.MaxPool1d(3)
        if in_dims != out_dims:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        else:
            self.downsample = False

    def forward(self, x):
        out = self.conv_first(x)
        out = self.batch_norm_first(out)
        out = self.prelu1(out)
        out = self.conv_second(out)
        out = self.batch_norm_second(out)
        if self.downsample:
            out += self.conv_downsample(x)
        else:
            out += x
        out = self.prelu2(out)
        return self.maxpooling(out)
