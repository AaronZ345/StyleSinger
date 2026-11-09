# StyleSinger: Style Transfer for Out-of-Domain Singing Voice Synthesis

Yu Zhang, Rongjie Huang, Ruiqi Li, JinZheng He, Yan Xia, Feiyang Chen, Xinyu Duan, Baoxing Huai, Zhou Zhao\
*Zhejiang University, Huawei Cloud*

[AAAI 2024](https://aaai.org/aaai-24-conference/)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2312.10741)
[![Demo](https://img.shields.io/badge/🚀%20Demo%20Page-blue)](https://aaronz345.github.io/StyleSingerDemo/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?label=Model)](https://huggingface.co/AaronZ345/StyleSinger)
[![GitHub Stars](https://img.shields.io/github/stars/AaronZ345/StyleSinger?style=social&label=GitHub+Stars)](https://github.com/AaronZ345/StyleSinger)

[![知乎](https://img.shields.io/badge/-知乎-000000?logo=zhihu&logoColor=0084FF)](https://zhuanlan.zhihu.com/p/775792127)

> **Start here** · [Quick start](#quick-start) · [Checkpoints](#checkpoints)\
> **Reference** · [Input format](#input-format) · [Training](docs/training.md) · [Citation](#citations)

StyleSinger transfers styles from reference singing to a target phoneme sequence and musical score. This repository includes the implementation and pretrained checkpoints.

| Released model | Requirements |
| --- | --- |
| Checkpoint language | Chinese |
| Reference audio | 48 kHz singing |
| Output | 48 kHz singing waveform |
| Runtime | Python 3.8, PyTorch 1.9.0 + CUDA 11.1 |

The released checkpoint supports **Chinese only**. For other languages, train a model using [GTSinger](https://github.com/AaronZ345/GTSinger) or your own dataset and matching phoneme vocabulary.

## Quick start

The example requires an NVIDIA GPU, the checkpoints below, and a reference recording. Run the commands from the repository root after preparing those files.

### 1. Install the environment

```bash
git clone https://github.com/AaronZ345/StyleSinger.git
cd StyleSinger
conda create -n stylesinger python=3.8 -y
conda activate stylesinger
python -m pip install -r requirements.txt \
  --find-links https://download.pytorch.org/whl/torch_stable.html
```

The requirements preserve the original model environment. The PyTorch wheel index supplies the pinned CUDA build.

### 2. Prepare the checkpoint and input

Download the `checkpoints/` directory from [Hugging Face](https://huggingface.co/AaronZ345/StyleSinger/tree/main/checkpoints) into the repository root. Place a copy of [ZH_checkpoint_phone_set.json](ZH_checkpoint_phone_set.json) at `data/processed/style/phone_set.json` for the released checkpoint.

Edit the `inp` dictionary in [StyleSingerInfer.example_run](inference/StyleSinger.py) with your reference path and target phoneme/score sequence. The included reference path is an example; supply your own audio file. Check the paths in [egs/stylesinger.yaml](egs/stylesinger.yaml).

### 3. Generate a sample

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python inference/StyleSinger.py \
  --config egs/stylesinger.yaml --exp_name checkpoints/StyleSinger
```

The example writes `infer_out/test.wav`. Set `CUDA_VISIBLE_DEVICES` to the existing GPU you intend to use.

## Checkpoints

| Component | Local path | Role |
| --- | --- | --- |
| StyleSinger | `checkpoints/StyleSinger/` | Acoustic model |
| HIFI-GAN | `checkpoints/hifigan/` | Vocoder checkpoint and `config.yaml` |
| Emotion encoder | `checkpoints/global.pt` | Global emotion representation |

[Hugging Face](https://huggingface.co/AaronZ345/StyleSinger) · [Google Drive mirror](https://drive.google.com/drive/folders/1C0Lp45EWFgcy7F3kGtU9s1wnyA8Nytbd?usp=sharing)

## Input format

| Field in `inp` | Meaning |
| --- | --- |
| `name` | Sample identifier |
| `ph` | Target phonemes in the checkpoint vocabulary |
| `note` | Target note for each phoneme |
| `note_dur` | Note duration for each phoneme |
| `note_type` | Rest: 1; lyric: 2; slur: 3 |
| `ref_audio` | Path to the 48 kHz reference singing recording |

Keep the phoneme and score arrays aligned. When using Chinese GTSinger annotations with this checkpoint, remove `_zh` from phonemes, map `<AP>` to `breathe`, and map `<SP>` to `_NONE`. Use the supplied [checkpoint vocabulary](ZH_checkpoint_phone_set.json), not a different training vocabulary.

## Method and training

The Residual Style Adaptor (RSA) uses residual quantization to represent reference styles. Uncertainty Modeling Layer Normalization (UMLN) perturbs style information in the content representation during training. See the [paper](https://ojs.aaai.org/index.php/AAAI/article/view/29932) for the architecture and zero-shot style-transfer comparisons.

The [training guide](docs/training.md) covers GTSinger metadata, train/validation/test splits, emotion-encoder preparation, binarization, training, and inference with your own model.

<details>
<summary>Release history</summary>

- September 2024: released the full [GTSinger](https://github.com/AaronZ345/GTSinger) dataset.
- May 2024: released StyleSinger code and checkpoints.
- December 2023: accepted to AAAI 2024.

</details>

## Acknowledgements

This implementation uses parts of the code from the following Github repos:
[GenerSpeech](https://github.com/Rongjiehuang/GenerSpeech),
[NATSpeech](https://github.com/NATSpeech/NATSpeech),
[ProDiff](https://github.com/Rongjiehuang/ProDiff),
[DiffSinger](https://github.com/MoonInTheRiver/DiffSinger)
as described in our code.

## Citations ##

If you find this code useful in your research, please cite our work:
```bib
@inproceedings{zhang2024stylesinger,
  title={StyleSinger: Style Transfer for Out-of-Domain Singing Voice Synthesis},
  author={Zhang, Yu and Huang, Rongjie and Li, Ruiqi and He, JinZheng and Xia, Yan and Chen, Feiyang and Duan, Xinyu and Huai, Baoxing and Zhao, Zhou},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={17},
  pages={19597--19605},
  year={2024}
}
```

## Disclaimer ##

Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's singing without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.
