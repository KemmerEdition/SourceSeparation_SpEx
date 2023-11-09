import json
import logging
import os
import glob
from glob import glob
import shutil
from pathlib import Path

import numpy as np
from speechbrain.utils.data_utils import download_file
from hw_asr.base.base_dataset import BaseDataset
from hw_asr.mixer.mixer import MixtureGenerator, speakers_list
from hw_asr.utils import ROOT_PATH

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class LibrispeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, mix=None, *args, **kwargs):
        assert part in URL_LINKS or part == 'train_all'

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = Path(data_dir)
        # if part == 'train_all':
        #     index = sum([self._get_or_load_index(part)
        #                  for part in URL_LINKS if 'train' in part], [])
        # else:
        index = self._get_or_load_index(part, mix)

        super().__init__(index, *args, **kwargs)

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))

    def _get_or_load_index(self, part, mix):
        index_path = mix.get("index_path", self._data_dir / f"{part}_index.json")
        index_path = Path(index_path)
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part, mix)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part, mix):
        index = []

        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        out_folder = mix.get("out_folder", self._data_dir / f"{part}-mixed")
        out_folder = Path(out_folder)
        if not out_folder.exists():
            self.mix_trainer(split_dir, out_folder, mix)

        ref_train = sorted(glob(os.path.join(out_folder, '*-ref.wav')))
        mix_train = sorted(glob(os.path.join(out_folder, '*-mixed.wav')))
        target_train = sorted(glob(os.path.join(out_folder, '*-target.wav')))
        ref_train, mix_train, target_train = np.asarray(ref_train, dtype=object), \
            np.asarray(mix_train, dtype=object), \
            np.asarray(target_train, dtype=object)
        id_sp = []
        for r in ref_train:
            id_sp.append(int(r.split('/')[-1].split('_')[0]))
        id_sp = np.asarray(id_sp)
        num_sp = np.asarray(id_sp).argsort()
        ref_train, mix_train, target_train, id_sp = ref_train[num_sp], mix_train[num_sp], target_train[num_sp], id_sp[num_sp]
        ids = list(range(len(id_sp)))

        for r, m, t, id_ in zip(ref_train, mix_train, target_train, ids):
            index.append(
                {
                    "reference": r,
                    "mix": m,
                    "target": t,
                    "speaker_id": id_
                }
            )
        return index

    def mix_trainer(self, split_dir, out_folder, mix):
        if mix is None:
            mix = {}
        test = mix.get("test", False)
        mixer_train = MixtureGenerator(
            speakers_files=speakers_list(split_dir, audioTemplate="*.flac"),
            out_folder=out_folder,
            nfiles=mix.get("nfiles", 10000),
            test=mix.get("test", False),
            randomState=42
        )

        # mixer_test = MixtureGenerator(
        #     speakers_files=speakers_list(audios_dir, audioTemplate="*.flac"),
        #     out_folder=out_folder,
        #     nfiles=mix.get("nfiles", 5000),
        #     test=mix.get("test", True),
        #     randomState=42
        # )

        if not test:
            mixer_train.generate_mixes(
                snr_levels=mix.get("snr_levels", [-5, 5]),
                num_workers=mix.get("num_workers", 2),
                update_steps=mix.get("update_steps", 100),
                trim_db=mix.get("trim_db", 20),
                vad_db=mix.get("vad_db", 20),
                audioLen=mix.get("audioLen", 3)
            )
        else:
            mixer_train.generate_mixes(
                snr_levels=mix.get("snr_levels", [0]),
                num_workers=mix.get("num_workers", 2),
                update_steps=mix.get("update_steps", 100),
                trim_db=mix.get("trim_db", None),
                vad_db=mix.get("vad_db", 20),
                audioLen=mix.get("audioLen", 3)
            )