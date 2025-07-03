# CineMyoPS: Myocardial Pathology Segmentation Network for Cine CMR Images


#### Introduction

####  Architecture

#### Installation Guide
- download our trained model from [baidu yun](https://pan.baidu.com/s/1ijuQaR0Ix6CE2Nu-TxareA?pwd=jrey)
- put them into ./outputs/nnunet/output/nnUNet/3d_fullres/Task025_Cine_Seg/TrainerV6WithoutIMG__nnUNetPlansv2.1/all

#### Usage Instructions

```bash
cd ./code

# Training Command
python ./Lascar_3_train.py 3d_fullres TrainerV6WithoutIMG Task025_Cine_Seg all

# Testing Command
python ./Lascar_4_test.py -tr TrainerV6WithoutIMG -i <your_input_path> -o <your_output_path> -t 025 --chk model_best --overwrite_existing --fold all
```

# Thanks
CineMyoPS is implemented based on [nnUNet](https://github.com/MIC-DKFZ/nnUNet) framework.
