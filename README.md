# SAFL: A Self-Attention Scene Text Recognizer with Focal Loss

This repository implements the SAFL in pytorch. 

The research paper can be found [here](https://ieeexplore.ieee.org/abstract/document/8395027/).

![SAFL Overview](overview.png)

## Installation

```
conda env create -f environment.yml
```

## Train

```
bash scripts/stn_att_rec.sh
```

## Test

You can test with .lmdb files by
```
bash scripts/main_test_all.sh
```
Or test with single image by
```
bash scripts/main_test_image.sh
```

## Data preparation

We give an example to construct your own datasets. Details please refer to `tools/create_svtp_lmdb.py`.

## Citation

If you find this project helpful for your research, please cite the following papers:
