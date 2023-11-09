# Don't forget to support cases when target_text == ''
import numpy as np

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
    alpha = (target * est).sum() / np.linalg.norm(target) ** 2
    return 20 * np.log10(np.linalg.norm(alpha * target) / (np.linalg.norm(alpha * target - est) + 1e-6) + 1e-6)
