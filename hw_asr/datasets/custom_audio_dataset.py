import logging
from pathlib import Path

from hw_asr.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomAudioDataset(BaseDataset):
    def __init__(self, data, *args, **kwargs):
        index = data
        for entry in data:
            assert "mix" in entry
            assert Path(entry["mix"]).exists(), f"Path {entry['mix']} doesn't exist"
            assert "reference" in entry
            assert Path(entry["reference"]).exists(), f"Path {entry['reference']} doesn't exist"
            assert "target" in entry
            assert Path(entry["target"]).exists(), f"Path {entry['target']} doesn't exist"
            assert "speaker_id" in entry
            entry["mix"] = str(Path(entry["mix"]).absolute().resolve())
            entry["reference"] = str(Path(entry["reference"]).absolute().resolve())
            entry["target"] = str(Path(entry["target"]).absolute().resolve())
            
        super().__init__(index, *args, **kwargs)
