
## Scripts to generate dataset 

### 1. Rectify the SfM models

Megadepth has 2 raw folders: 

1. MegaDepth_v1: contains the resized images and depth maps.
2. MegaDepth_v1_SfM: contains the raw images and SfM models.

Notice that the raw SfM models inside MegaDepth_v1_SfM use SIMPLE_RADIAL(colmap) camera, we need to rectify the SfM model to PINHOLE. Because the resized images and depth maps inside MegaDepth_v1 were actually rectified.

Use `rectify_megadepth.py` to generates rectified models. **Please specify the local path inside the script.** It will generate an rectified SfM model alongside with original model. For example, if the original model is at `/MegaDepth_v1_SfM/0000/sparse/manhattan/0`, then the rectified model will be at `/MegaDepth_v1_SfM/0000/sparse/manhattan/0_rectified/sparse`.

### 2. Valid list

Some depth maps provided by Megadepth are actually semantic depth, which is an ordering mask. We checked all .h5 files and filter out any depth map that the minimum depth value is less than 0.

Use `prepare_megadepth_valid_list.py` to generate the valid list, or use the provided valid list(`megadepth_valid_list.json`).

### 3. Train/val/test split

We use scenes [0000, 0240] **EXCEPT** scene 0204 as training split, we use scene 0204 as the validation split, and the rest as the test split.

Use `prepare_megadepth_split.py` to generate the split, or use the provided split files(`megadepth_train.json` and `megadepth_val.json`).

### 4. Sequences control

We use another json file to control the sequences we want to use during training. It allows us to use a smaller sequence to debug, and remove some unwanted sequences.

Notice that in the final training(current version), we remove the overlapping scenes with IMW dataset as mentioned [here](https://www.cs.ubc.ca/research/image-matching-challenge/2020/submit/).

Build you own sequence control json, or use the provided ones(`200_megadepth.json` for default training, and `debug_megadepth.json` for debugging).

### 5. Distance matrix

Under each `denseX` folder inside folder `MegaDepth_v1` , we add a `dist_mat` folder, and inside the folder is the `dist_mat.npy` which represents the distance matrix.

For example: `/MegaDepth_v1/phoenix/S6/zl548/MegaDepth_v1/0000/dense0/dist_mat/dist_mat.npy`

The size of the distance matrix is N by N, where N is the number of images with **valid** depth.

The index of the matrix aligns with the order in the "images.txt", thus we require at least python 3.7 which uses ordered dictionary as default.

Use `prepare_nn_distance_mat.py` to generate the distance matrix, or use the provided distance matrix([link](https://www.cs.ubc.ca/research/kmyi_data/files/2021/cotr/MegaDepth_v1.zip)).
