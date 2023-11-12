import logging
from pathlib import Path

from hw_asr.datasets.custom_audio_dataset import CustomAudioDataset
import os
from glob import glob

logger = logging.getLogger(__name__)


def path_file(files):
    return {os.path.basename(path).split("-")[0]: path for path in files}


class CustomDirAudioDataset(CustomAudioDataset):
    def __init__(self, mix_dir, ref_dir, target_dir, *args, **kwargs):
        data = []

        mixes = path_file(glob(os.path.join(mix_dir, '*-mixed.wav')))
        refs = path_file(glob(os.path.join(ref_dir, '*-ref.wav')))
        targets = path_file(glob(os.path.join(target_dir, '*-target.wav')))

        for key in (mixes.keys() & refs.keys() & targets.keys()):
            data.append({
                "mix": mixes[key],
                "reference": refs[key],
                "target": targets[key],
                "speaker_id": 0 
            })
        
        super().__init__(data, *args, **kwargs)
