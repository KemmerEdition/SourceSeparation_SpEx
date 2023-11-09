import torch.nn as nn
from hw_asr.metric.utils import si_sdr


class SpexLoss(nn.Module):
    def __init__(self, param_a=0.5, param_b=0.1, param_c=0.1):
        super().__init__()
        self.param_a = param_a
        self.param_b = param_b
        self.param_c = param_c
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, short, middle, long, speaker_pred, speaker_id, target, **kwargs):

        short = short - short.mean(dim=-1, keepdim=True)
        middle = middle - middle.mean(dim=-1, keepdim=True)
        long = long - long.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)

        compute_loss = (-(1- self.param_b - self.param_c) * si_sdr(short, target).sum()
                        - self.param_b * si_sdr(middle, target).sum()
                        - self.param_c * si_sdr(long, target).sum()) / short.shape[0]

        compute_cross_entropy = self.cross_entropy(speaker_pred, speaker_id.to(speaker_pred.device))
        loss_result = compute_loss + compute_cross_entropy * self.param_a
        return loss_result
