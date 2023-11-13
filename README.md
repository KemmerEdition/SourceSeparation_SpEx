# Source Separation (SPEX+) project

## Installation guide

### Part 1. Create (or download) a dataset for this work.
In my report I described the process of generating and some difficulties. If you do not want to generate mixes by yourself download zip from here:
#### Train-clean-100-mixed part:
```shell
https://drive.google.com/file/d/11wfHRkvOQqIfmOpnQB8stwpeRgtJbJyX/view
   ```
#### Test-clean-mixed part for validation respectively:
```shell
https://drive.google.com/file/d/1LIPaY_pxXzHRDiyzpQvkt7a2N_IQSz8X/view
   ```
#### If you use Kaggle GPU for learning - cry and use my public dataset there:
```shell
https://www.kaggle.com/datasets/anastasiakemmer/librispeech-mixed
   ```
### Part 2 (If you are still alive).
#### Clone repository
   ```shell
   git clone https://github.com/KemmerEdition/HW-2-SS.git
   ```
#### Maybe then you need to change directory (for example if you use kaggle)

   ```shell
   cd /kaggle/working/HW-2-SS
   ```
#### Download requirements and checkpoint of my model
   ```shell
   pip install -r requirements.txt
   pip install pesq
   ```
   ```shell
   !conda install -y gdown
   !gdown --id 1AbdteSPpitDptIksoQQtQbuaytXNJNpy
   ```
## Train
   ```shell
   python -m train \
      -c hw_asr/configs/train_bs3_30k.json
   ```
## Test
#### For using this code you need to upload test_data directory in workspace, like it'd done here (directory should contain mix, refs, targets folders)
   ```shell
   python -m test \
      -c hw_asr/tests/config.json \
      -r checkpoint-epoch5.pth \
      -t test_data_2 \
      -o test_result.json
   ```

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
