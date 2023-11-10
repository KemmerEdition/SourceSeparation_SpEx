from hw_asr.base.base_metric import BaseMetric
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
import torch
from torch import Tensor


class PesQ(BaseMetric):
    def __init__(self, fs=16000, mode="wb", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pesq = PerceptualEvaluationSpeechQuality(fs, mode)

    def __call__(self, short: Tensor, target: Tensor, **kwargs):
        metric_ = self.pesq.to(short.device)
        metric_result_norm = 20 * short / short.norm(-1, keepdim=True)
        metric_result = metric_(metric_result_norm, target).mean().item()
        return metric_result
