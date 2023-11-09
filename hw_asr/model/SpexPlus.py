import torch
import torch.nn as nn
import torch.nn.functional as F
from hw_asr.model.EncoderDecoder import SpexEncoder, SpexDecoder, LayerNorm, TCNBlocksSpex, ResNetSpex


class SpexPlus(nn.Module):
    def __init__(self,
                 short_size=20, middle_size=80, long_size=160,
                 speech_encoder_out_channels=256, speaker_encoder_hidden_channels=256,
                 tcn_input=256, tcn_hidden=512, n_tcn_block=8, kernel_size=3, speakers_counter=100
                 ):
        super().__init__()

        self.encode_1d = SpexEncoder(short_size, middle_size, long_size, speech_encoder_out_channels)
        self.short_size = short_size
        self.LN = LayerNorm(3 * speech_encoder_out_channels)
        self.conv = nn.Conv1d(3 * speech_encoder_out_channels, tcn_input, 1)
        self.part_res = nn.Sequential(LayerNorm(3 * speech_encoder_out_channels),
                                      nn.Conv1d(3 * speech_encoder_out_channels, tcn_input, 1),
                                      ResNetSpex(tcn_input, tcn_input),
                                      ResNetSpex(tcn_input, tcn_hidden),
                                      ResNetSpex(tcn_hidden, tcn_hidden),
                                      nn.Conv1d(tcn_hidden, speaker_encoder_hidden_channels, 1))

        part_tcn = [TCNBlocksSpex(in_channels=tcn_input, conv_channels=tcn_hidden,
                                  sp_channels=speaker_encoder_hidden_channels, kernel_size=kernel_size,
                                  num_blocks=n_tcn_block) for i in range(4)]
        self.part_tcn = nn.ModuleList(part_tcn)

        post_tcn = [nn.Sequential(nn.Conv1d(tcn_input, speech_encoder_out_channels, 1),
                                  nn.ReLU()) for i in range(3)]
        self.post_tcn = nn.ModuleList(post_tcn)

        self.decoder_1d = SpexDecoder(short_size, middle_size, long_size, speech_encoder_out_channels)
        self.linear = nn.Linear(speaker_encoder_hidden_channels, speakers_counter)

    def forward(self, x, sp_emb, emb_len, **kwargs):
        length = x.shape[-1]
        mix_mask_list = []
        mix_encoded = self.encode_1d(x)
        encoded = torch.cat(self.encode_1d(x), 1)
        x = self.LN(encoded)
        x = self.conv(x)
        new_length = (emb_len - self.short_size) // (self.short_size // 2) + 1
        new_length = ((new_length // 3) // 3) // 3
        new_length = new_length.float()
        sp_emb_encoded = torch.cat(self.encode_1d(sp_emb), 1)
        sp_emb = self.part_res(sp_emb_encoded).sum(-1, keepdeep=True) / new_length[:, None, None]
        for tcn in self.part_tcn:
            x = tcn(x, sp_emb)
        for i, v in zip(self.post_tcn, mix_encoded):
            mix_mask_list.append(v * i(x))
        dec_short, dec_mid, dec_long = self.decoder_1d(*mix_mask_list)
        preds = self.linear(sp_emb.squeeze(-1))
        dec_short = F.pad(dec_short, (0, length - dec_short.shape[-1]))
        return {"speaker_pred": preds, "short": dec_short, "middle": dec_mid[:, :, :length], "long": dec_long[:, :, :length]}
