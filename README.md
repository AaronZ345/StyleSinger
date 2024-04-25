# StyleSinger: Style Transfer for Out-of-Domain Singing Voice Synthesis

#### Yu Zhang, Rongjie Huang, Ruiqi Li, JinZheng He, Yan Xia, Feiyang Chen, Xinyu Duan, Baoxing Huai, Zhou Zhao | Zhejiang University, Huawei Cloud

PyTorch Implementation of [StyleSinger (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29932/31629): Style Transfer for Out-of-Domain Singing Voice Synthesis.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2312.10741)

We provide our implementation and pre-trained models in this repository.

Visit our [demo page](https://stylesinger.github.io/) for audio samples.

## News
- April, 2024: StyleSinger released at GitHub.
- December, 2023: StyleSinger accepted at AAAI 2024.

## Quick Started
We provide an example of how you can generate high-quality samples using AlignSTS.

### Pre-trained Models
You can use the pre-trained models we provide [here](). Details of each folder are as follows:

| Model       |  Description                                                              | 
|-------------|------------------|--------------------------------------------------------------------------|
| StyleSinger |  Acousitic model [(config)]() |
| HIFI-GAN    |  Neural Vocoder                                                           |
| Encoder     |  Emotion Encoder                                                   |

### Dependencies

A suitable [conda](https://conda.io/) environment named `stylesinger` can be created
and activated with:

```
conda create -n stylesinger python=3.8
conda install --yes --file requirements.txt
conda activate stylesinger
```
