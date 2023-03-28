# ABCD : Arbitrary Bitwise Coefficient for De-quantization
This repository contains the official implementation for ABCD introduced in the following paper:

ABCD : Arbitrary Bit-wise Coefficient for De-quantization (CVPR 2023)

### DEMO 
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
