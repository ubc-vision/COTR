# COTR: Correspondence Transformer for Matching Across Images

This repository contains the inference code for COTR. We plan to release the training code in the future.
COTR establishes correspondence in a functional and end-to-end fashion. It solves dense and sparse correspondence problem in the same framework.

[arXiv](https://arxiv.org/abs/2103.14167)

## Demos

Check out our demo video at [here](https://jiangwei221.github.io/vids/cotr/README.html).

### 1. Install environment

Our implementation is based on PyTorch. Install the conda environment by: `conda env create -f environment.yml`.

Activate the environment by: `conda activate cotr_env`.

Notice that we use `scipy=1.2.1` .


### 2. Download the pretrained weights

Down load the pretrained weights at [here](https://www.cs.ubc.ca/research/kmyi_data/files/2021/cotr/default.zip). Extract in to `./out`, such that the weights file is at `/out/default/checkpoint.pth.tar`.

### 3. Single image pair demo

```python demo_single_pair.py --load_weights="default"```

Example sparse output:

<p align="center">
  <img src="./sample_data/imgs/sparse_output.png" height="400">
</p>

Example dense output with triangulation:

<p align="center">
  <img src="./sample_data/imgs/dense_output.png" height="200">
</p>

**Note:** This example uses 10K valid sparse correspondences to densify.

### 4. Facial landmarks demo

`python demo_face.py --load_weights="default"`

Example:

<p align="center">
  <img src="./sample_data/imgs/face_output.png" height="200">
</p>

### 5. Homography demo

`python demo_homography.py --load_weights="default"`

<p align="center">
  <img src="./sample_data/imgs/paint_output.png" height="300">
</p>

### 6. Guided matching demo

`python demo_guided_matching.py --load_weights="default"`

<p align="center">
  <img src="./sample_data/imgs/guided_matching_output.png" height="400">
</p>

### 7. Two view reconstruction demo

Note: this demo uses both known camera intrinsic and extrinsic.
`python demo_reconstruction.py --load_weights="default" --max_corrs=2048 --faster_infer=yes`

<p align="center">
  <img src="./sample_data/imgs/recon_output.png" height="250">
</p>

## Faster Inference

We added a faster inference engine.
The idea is that for each network invocation, we want to solve more queries. We search for nearby queries and group them on the fly.
*Note: Faster inference engine has slightly worse spatial accuracy.*
Guided matching demo now supports faster inference.
The time consumption for default inference engine is ~216s, and the time consumption for faster inference engine is ~79s, on 1080Ti.
Try `python demo_guided_matching.py --load_weights="default" --faster_infer=yes`.

## Citation

If you use this code in your research, cite the paper:

```
@article{jiang2021cotr,
  title={{COTR: Correspondence Transformer for Matching Across Images}},
  author={Wei Jiang and Eduard Trulls and Jan Hosang and Andrea Tagliasacchi and Kwang Moo Yi},
  booktitle={arXiv preprint},
  publisher_page={https://arxiv.org/abs/2103.14167},
  year={2021}
}
```
