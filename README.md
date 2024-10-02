# StyleSinger: Style Transfer for Out-of-Domain Singing Voice Synthesis

#### Yu Zhang, Rongjie Huang, Ruiqi Li, JinZheng He, Yan Xia, Feiyang Chen, Xinyu Duan, Baoxing Huai, Zhou Zhao | Zhejiang University, Huawei Cloud

PyTorch Implementation of [StyleSinger (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29932/31629): Style Transfer for Out-of-Domain Singing Voice Synthesis.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2312.10741)

We provide our implementation and pre-trained models in this repository.

Visit our [demo page](https://stylesinger.github.io/) for audio samples.

### Pre-trained Models
You can use the pre-trained models we provide [here](https://drive.google.com/drive/folders/1C0Lp45EWFgcy7F3kGtU9s1wnyA8Nytbd?usp=sharing).**Notably，this checkpoint only support Chinese! You should train your own model based on GTSinger for multilingual style transfer!** Details of each folder are as follows:

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

## Inference towards style transfer of custom timbre and style for Chinese singing voices

Here we provide a speech synthesis pipeline using StyleSinger. 

1. Prepare **StyleSinger** (acoustic model): Download and put checkpoint at `checkpoints/StyleSinger` 
2. Prepare **HIFI-GAN** (neural vocoder): Download and put checkpoint at `checkpoints/hifigan`
3. Prepare **Emotion Encoder**: Download and put checkpoint at `checkpoints/global.pt`
4. Prepare **dataset**: Download and put statistical files at `data/binary/test_set`
5. Prepare **reference information**: Provide a reference_audio (48k) and input target ph, target note for each ph, target note_dur for each ph, target note_type for each ph (rest: 1, lyric: 2, slur: 3), and reference audio path. Input these information in `Inference/StyleSinger.py`.

```bash
CUDA_VISIBLE_DEVICES=$GPU python inference/StyleSinger.py --config egs/stylesinger.yaml  --exp_name checkpoints/StyleSinger
```

Generated wav files are saved in `infer_out` by default.<br>

# Train your own model based on GTSinger for multilingual style transfer

### Data Preparation 

1. Prepare your own singing dataset or download [GTSinger](https://github.com/GTSinger/GTSinger) (Note: we provide `metadata.json` and `phone_set.json` in GTSinger)
2. Put `metadata.json` (including ph, word, item_name, ph_durs, wav_fn, singer, ep_pitches, ep_notedurs, ep_types for each singing voice) and `phone_set.json` (all phonemes of your dictionary) in `data/processed/style`
3. Set `processed_data_dir`, `binary_data_dir`,`valid_prefixes`, `test_prefixes` in the [config](./egs/stylesinger.yaml).
4. Download the global emotion encoder to `emotion_encoder_path` (for Chiense only) or train your own global emotion encoder refers to [Emotion Encoder](https://github.com/Rongjiehuang/GenerSpeech/tree/encoder) based on emotion annotations in GTSinger. 
5. Preprocess Dataset 

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

### Quick Inference

We provide a mini-set of test samples to demonstrate StyleSinger in [here](https://drive.google.com/drive/folders/1O4pn7UeLzLGjL89nHd7F-rSQybhUCzrA?usp=sharing). Specifically, we provide samples of statistical files which is for faster IO. Please download the statistical files at `data/binary/style/`, while the WAV files are for listening.

Run
```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config egs/stylesinger.yaml  --exp_name StyleSinger --infer
```

You will find outputs in `checkpoints/StyleSinger/generated_320000_/wavs`, where [Ref] indicates ground truth mel results and [SVS] indicates predicted results.

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
