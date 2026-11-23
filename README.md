# CamVid road-scene segmentation

How much network depth does a model need to tell road, sidewalk and buildings apart? I adapted the width and depth search idea from ColabNAS to an encoder-decoder with skip connections. My original experiments also varied the activation function. This repository contains a small training script for comparing individual configurations.

## Run a comparison

Use Python 3.9 and the RGB-mask version of CamVid. Point the script at a directory containing `class_dict.csv` (columns `name,r,g,b`), `train/`, `train_labels/`, `val/` and `val_labels/`. For example, the mask for `0001TP_009210.png` is `0001TP_009210_L.png`. The order of the palette rows determines the class IDs, including the supplied void class.

```sh
python -m pip install -r requirements.txt
python segmentation.py CamVid --width 16 --depth 2 --activation relu
python segmentation.py CamVid --width 16 --depth 2 --activation leaky_relu
```

Each command trains one candidate and prints its parameter count and best validation pixel accuracy. Keeping the width, depth, epochs and split fixed lets you compare activations. Pixel accuracy gives common classes more weight. Run `python -m unittest -v` for the small file-pairing, mask and model checks.

## Evaluation

The loader pairs frames by ID and resizes masks with nearest-neighbor interpolation. Training and validation come from separate directories. The original notebook sampled training images for its exploratory IoU calculation, so there is no held-out IoU result reported here. Supply the dataset locally. Images and trained weights are outside this repository.

## References

- [CamVid dataset paper](https://doi.org/10.1016/j.patrec.2008.04.005) by Brostow, Fauqueur and Cipolla.
- [ColabNAS](https://github.com/AndreaMattiaGaravagno/ColabNAS) by Garavagno, Leonardis and Frisoli inspired the architecture comparison. The road-scene segmentation experiments are my adaptation. See the [paper](https://doi.org/10.1016/j.future.2023.11.003) and the upstream license in `COLABNAS-LICENSE.txt`.
