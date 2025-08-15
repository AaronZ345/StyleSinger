# StyleSinger: Style Transfer for Out-of-Domain Singing Voice Synthesis

#### Yu Zhang, Rongjie Huang, Ruiqi Li, JinZheng He, Yan Xia, Feiyang Chen, Xinyu Duan, Baoxing Huai, Zhou Zhao | Zhejiang University, Huawei Cloud

PyTorch Implementation of [StyleSinger (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29932): Style Transfer for Out-of-Domain Singing Voice Synthesis.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2312.10741)
[![Demo](https://img.shields.io/badge/🚀%20Demo%20Page-blue)](https://aaronz345.github.io/StyleSingerDemo/)
[![zhihu](https://img.shields.io/badge/-知乎-000000?logo=zhihu&logoColor=0084FF)](https://zhuanlan.zhihu.com/p/775792127)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?label=Model)](https://huggingface.co/AaronZ345/StyleSinger)
[![GitHub Stars](https://img.shields.io/github/stars/AaronZ345/StyleSinger?style=social&label=GitHub+Stars)](https://github.com/AaronZ345/StyleSinger)

We provide our implementation and pre-trained models in this repository.

Visit our [demo page](https://aaronz345.github.io/StyleSingerDemo/) for audio samples.

## News
- 2024.09: We released the full dataset of [GTSinger](https://github.com/AaronZ345/GTSinger)!
- 2024.05: We released the code and checkpoints of StyleSinger!
- 2023.12: StyleSinger is accepted by AAAI 2024!

## Key Features

- We present **StyleSinger**, the first singing voice synthesis model for zero-shot style transfer of out-of-domain reference samples. StyleSinger excels in generating exceptional singing voices with unseen styles derived from reference singing voice samples.
- We propose the **Residual Style Adaptor (RSA)**, which uses a residual quantization model to meticulously capture diverse style characteristics in reference samples.
- We introduce the **Uncertainty Modeling Layer Normalization (UMLN)** to perturb the style information in the content representation during the training phase, and thus enhance the model generalization of StyleSinger.
- Extensive experiments in **zero-shot style transfer** show that StyleSinger exhibits superior audio quality and similarity compared with baseline models.

## Quick Start
We provide an example of how you can generate high-fidelity samples using StyleSinger.

To try on your own dataset or GTSinger, simply clone this repo on your local machine provided with NVIDIA GPU + CUDA cuDNN and follow the below instructions.

### Pre-trained Models
You can use all pre-trained models we provide on [HuggingFace](https://huggingface.co/AaronZ345/StyleSinger) or [Google Drive](https://drive.google.com/drive/folders/1C0Lp45EWFgcy7F3kGtU9s1wnyA8Nytbd?usp=sharing). **Notably, this StyleSinger checkpoint only supports Chinese! You should train your own model based on GTSinger for multilingual style transfer!** Details of each folder are as follows:

| Model       |  Description                                                              | 
|-------------|--------------------------------------------------------------------------|
| StyleSinger |  Acousitic model [(config)](./egs/stylesinger.yaml) |
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

### Multi-GPU

By default, this implementation uses as many GPUs in parallel as returned by `torch.cuda.device_count()`. 
You can specify which GPUs to use by setting the `CUDA_DEVICES_AVAILABLE` environment variable before running the training module.

## Inference for Chinese singing voices

Here we provide a speech synthesis pipeline using StyleSinger. 

1. Prepare **StyleSinger** (acoustic model): Download and put checkpoint at `checkpoints/StyleSinger`.
2. Prepare **HIFI-GAN** (neural vocoder): Download and put checkpoint at `checkpoints/hifigan`.
3. Prepare **Emotion Encoder**: Download and put checkpoint at `checkpoints/global.pt`.
4. Prepare **reference information**: Provide a reference_audio (48k) and input target ph, target note for each ph, target note_dur for each ph, target note_type for each ph (rest: 1, lyric: 2, slur: 3), and reference audio path. Input these information in `Inference/StyleSinger.py`. **Notably, if you want to use Chinese data in GTSinger to infer this Chinese checkpoint, refer to [phone_set](./ZH_checkpoint_phone_set.json), you have to delete _zh in each ph of GTSinger, and change \<AP\> to breathe, \<SP\> to _NONE!**
5. Infer for style transfer:

```bash
rm -rf ./checkpoints/checkpoints
CUDA_VISIBLE_DEVICES=$GPU python inference/StyleSinger.py --config egs/stylesinger.yaml  --exp_name checkpoints/StyleSinger
```

Generated wav files are saved in `infer_out` by default.<br>

## Train your own model based on GTSinger

### Data Preparation 

1. Prepare your own singing dataset or download [GTSinger](https://github.com/AaronZ345/GTSinger).
2. Put `metadata.json` (including ph, word, item_name, ph_durs, wav_fn, singer, ep_pitches, ep_notedurs, ep_types for each singing voice) and `phone_set.json` (all phonemes of your dictionary) in `data/processed/style` **(Note: we provide `metadata.json` and `phone_set.json` in GTSinger, but you need to change the wav_fn of each wav in `metadata.json` to your own absolute path)**.
3. Set `processed_data_dir` (`data/processed/style`), `binary_data_dir`, `valid_prefixes` (list of parts of item names, like `["Chinese#ZH-Alto-1#Mixed_Voice_and_Falsetto#一次就好"]`), `test_prefixes` in the [config](./egs/stylesinger.yaml).
4. Download the global emotion encoder to `emotion_encoder_path` (training on Chinese only) or train your own global emotion encoder referring to [Emotion Encoder](https://github.com/Rongjiehuang/GenerSpeech/tree/encoder) based on emotion annotations in GTSinger. 
5. Preprocess Dataset: 

```bash
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=$GPU python data_gen/tts/bin/binarize.py --config egs/stylesinger.yaml
```

### Training StyleSinger

```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config egs/stylesinger.yaml  --exp_name StyleSinger --reset
```

### Inference using StyleSinger

```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config egs/stylesinger.yaml  --exp_name StyleSinger --infer
```

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

 ![visitors](https://visitor-badge.laobi.icu/badge?page_id=AaronZ345/StyleSinger)
