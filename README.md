# ABCD : Arbitrary Bitwise Coefficient for De-quantization
This repository contains the official implementation for ABCD introduced in the following paper:

[ABCD : Arbitrary Bit-wise Coefficient for De-quantization (CVPR 2023)](https://ipl.dgist.ac.kr/ABCD_cvpr23.pdf)

## DEMO 



https://user-images.githubusercontent.com/92767986/228137480-6177e124-f610-4002-8861-481165e0ac3b.mp4

Our ABCD reconstruct randomly quantized images with single training!

## Installation
Our code is based on Ubuntu 20.04, pytorch 1.10.0, CUDA 11.3 (NVIDIA RTX 3090 24GB, sm86) and python 3.6.

For enviorment setup, we recommend to use [conda](https://www.anaconda.com/distribution/) for installation:

```
conda env create --file environment.yaml
conda activate abcd
```
## Datasets

Our train and valid sets follows the prior work [Punnappurath et. al.](https://github.com/abhijithpunnappurath/a-little-bit-more/tree/master/download_data_and_test), sampled from [MIT-Adove FiveK](https://data.csail.mit.edu/graphics/fivek/) and [Sintel](https://media.xiph.org/sintel/sintel-1k-png16/) dataset. 

For the benchmark part in Tab 1, we tested [Kodak](https://r0k.us/graphics/kodak/), [TESTIMAGES1200](https://testimages.org/) (\B01C00 folder), and [ESPLv2](http://signal.ece.utexas.edu/~bevans/synthetic/) datasets.


## Train
The basic train code is : 
```
python train.py --config configs/train_ABCD/SomeConfigs.yaml --gpu 0
```
We provide configurations of our main paper

**EDSR-ABCD** : `configs/train_ABCD/train_EDSR-ABCD.yaml`

**RDN-ABCD** : `configs/train_ABCD/train_RDN-ABCD.yaml` 

**SwinIR-ABCD** : `configs/train_ABCD/train_SwinIR-ABCD.yaml`

**EDSR-baseline_ABCD** : `configs/train_ABCD/train_EDSR-baseline_ABCD.yaml`

If you want to modify some configuration (e.g. the range of input bit-depth) modify `.yaml` files and run 
```
python train.py --config configs/train_ABCD/FancyConfiguration.yaml --gpu 0
```

### Pretrained Models

We provide pre-trained models of our ABCD. Test our models with test codes below.

Check the training time and required the number of GPU for training.

Model|Training time (# GPU)
:-:|:-:
[EDSR-ABCD](https://drive.google.com/file/d/1LAe1KUPe8MuOP_NRwBMfQ5W32o37Ln6W/view?usp=sharing)|65h (1 GPU)
[RDN-ABCD](https://drive.google.com/file/d/1tj7HiSpDxuHdEFQYG_EwWncDisfT_k88/view?usp=sharing)|82h (2 GPU)
[SwinIR-ABCD](https://drive.google.com/file/d/1zBGLttDMET7CQcj729sZyPKOVWpMyyMZ/view?usp=sharing)|130h (4 GPU)

We recommend trying ``EDSR-baseline_ABCD`` since it is lighter than models above.


## Test

To test our model, run the code below : 
```
python test.py --config configs/test_ABCD/abcd_test-16bits.yaml --model save/PATHS/MODEL.pth --LBD 4 --HBD 8 --gpu 0
```


**16-bit image dataset** : `/configs/test_ABCD/abcd_test-16bits.yaml` (e.g. MIT-5K, TESTIMAGES1200) 

**8-bit image dataset** : `/configs/test_ABCD/abcd_test-8bits.yaml` (e.g. Kodak) 

If you want to test another labels, you may change flag 'LBD' and 'HBD' to test your model.

(e.g. 3-bits to 12-bits ``--LBD 3 --HBD 12``, respectively)

For SwinIR based ABCD, test code needs additional flag ``--window 8`` 

You can save results with using the flag ``--save 1``.

## Acknowledgements

This code is built on [LIIF](https://github.com/yinboc/liif),[LTE](https://github.com/jaewon-lee-b/lte) and [SwinIR](https://github.com/JingyunLiang/SwinIR). We thank the authors for sharing their codes.
