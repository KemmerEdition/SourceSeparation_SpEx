# from typing import List
#
# import torch
# from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


class SiSdr(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sisdr = ScaleInvariantSignalDistortionRatio()

    def __call__(self, short, target, **kwargs):
        metric_result = self.sisdr.to(short.device)
        metric_result = metric_result(short, target)
        return metric_result
