[Back to StyleSinger](../README.md)

# Train StyleSinger on GTSinger

Run these commands from the repository root in the environment described in the main README. Set `GPU` to an existing device, for example `export GPU=0`.

## Data preparation 

1. Prepare your own singing dataset or download [GTSinger](https://github.com/AaronZ345/GTSinger).
2. Put `metadata.json` (including ph, word, item_name, ph_durs, wav_fn, singer, ep_pitches, ep_notedurs, ep_types for each singing voice) and `phone_set.json` (all phonemes of your dictionary) in `data/processed/style` **(Note: we provide `metadata.json` and `phone_set.json` in GTSinger, but you need to change the wav_fn of each wav in `metadata.json` to your own absolute path)**.
3. Set `processed_data_dir` (`data/processed/style`), `binary_data_dir`, `valid_prefixes` (list of parts of item names, like `["Chinese#ZH-Alto-1#Mixed_Voice_and_Falsetto#一次就好"]`), `test_prefixes` in the [config](../egs/stylesinger.yaml).
4. Download the global emotion encoder to `emotion_encoder_path` (training on Chinese only) or train your own global emotion encoder referring to [Emotion Encoder](https://github.com/Rongjiehuang/GenerSpeech/tree/encoder) based on emotion annotations in GTSinger. 
5. Preprocess Dataset: 

```bash
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=$GPU python data_gen/tts/bin/binarize.py --config egs/stylesinger.yaml
```

## Training

```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config egs/stylesinger.yaml  --exp_name StyleSinger --reset
```

## Inference

```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config egs/stylesinger.yaml  --exp_name StyleSinger --infer
```
