# COTR: Correspondence Transformer for Matching Across Images

This repository contains the inference code for COTR. We plan to release the training code in the future.
COTR establishes correspondence in a functional and end-to-end fashion. It solves dense and sparse correspondence problem in the same framework.

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
