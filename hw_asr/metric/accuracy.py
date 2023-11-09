from hw_asr.base.base_metric import BaseMetric
from torch import Tensor


class Accuracy(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, speaker_pred, speaker_id: Tensor, **kwargs):
        return (speaker_pred.argmax(dim=-1) == speaker_id).float().mean().item()
