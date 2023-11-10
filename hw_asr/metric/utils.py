# Don't forget to support cases when target_text == ''
import numpy as np
import torch

#
# def calc_cer(target_text, predicted_text) -> float:
#     # TODO: your code here
#     raise NotImplementedError()
#
#
# def calc_wer(target_text, predicted_text) -> float:
#     # TODO: your code here
#     raise NotImplementedError()


def si_sdr(est, target):
    alpha = (target * est).sum() / torch.linalg.norm(target) ** 2
    return 20 * torch.log10(torch.linalg.norm(alpha * target) / (torch.linalg.norm(alpha * target - est) + 1e-6) + 1e-6)
