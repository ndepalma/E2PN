
# E2PN: Efficient SE(3)-Equivariant Point Network (CVPR 2023)

[![Watch the video](https://img.youtube.com/vi/B4XDxd0h08I/hqdefault.jpg)](https://youtu.be/B4XDxd0h08I)


This repository contains the official code for the CVPR 2023 paper **E2PN: Efficient SE(3)-Equivariant Point Network** [[paper](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_E2PN_Efficient_SE3-Equivariant_Point_Network_CVPR_2023_paper.html)] [[poster and slides](https://cvpr2023.thecvf.com/virtual/2023/poster/21183)]. 


## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Experiments](#experiments)
4. [Running on RunPod](#running-on-runpod)
5. [Contacts](#contacts)

## Introduction

E2PN is a SE(3)-equivariant network architecture designed for deep point cloud analysis. It largely improves the efficiency in terms of memory consumption and running speed for equivariant point cloud feature learning. 

It mainly includes a SE(3)-equivariant convolution on $S^2\times \mathbb{R}^3$ which is a homogeneous space of SE(3), and a permutation layer to recover SE(3) information from the features in the homogeneous space. 

<!-- ![](https://github.com/nintendops/EPN_PointCloud/blob/main/media/spconv.png) -->



## Usage
Our code is developed based on an open-sourced existing work [EPN](https://github.com/nintendops/EPN_PointCloud). We use the same dependencies as the [EPN](https://github.com/nintendops/EPN_PointCloud) repository. 

The original code was written for Python 3.7, PyTorch 1.11 and CUDA 11.3, and needed a compiled CUDA extension. This version is pure PyTorch: it runs on Apple Silicon (MPS), CUDA or CPU with no build step. It has been tested with Python 3.14 and PyTorch 2.15 on macOS (MPS and CPU).

```
pip install -r requirements.txt
```

The run scripts add `vgtk` to the path themselves; `pip install -e vgtk` is optional.

The device is chosen automatically (MPS, then CUDA, then CPU). Use `--device cpu|mps|cuda` in the `experiment` group to override it, e.g. `python run_modelnet.py experiment --device cpu ...`.

**Logging.** Each run writes `log.txt`, `opt.json` and checkpoints under `--model-dir/<experiment-id>/model_<timestamp>/`. To also log loss, accuracy, learning rate and test metrics to Weights & Biases, run `wandb login` once and add `--wandb` (and optionally `--wandb-project NAME`, default `E2PN`) to the `experiment` group:

```
python run_modelnet.py experiment --experiment-id cls_s2 --wandb --model-dir PATH_TO_LOG -d PATH_TO_MODELNET40 \
model --flag permutation --kanchor 12 --feat-all-anchors --drop_xyz
```

The former CUDA kernels (ball query, furthest point sampling, gathering, anchor query) live in `vgtk/vgtk/ops.py`. To check them against reference implementations of the original kernels, and to check the rotation anchors:

```
python -m pytest vgtk/tests
```

## Experiments

**Datasets**

The rotated Modelnet40 point cloud dataset is generated from the [Aligned Modelnet40 subset](https://github.com/lmb-freiburg/orion) and can be downloaded using this [link](https://drive.google.com/file/d/1xRoYjz2KCwkyIPf21E-WKIZkjLYabPgJ/view?usp=sharing).

The original 3DMatch training and evaluation dataset can be found [here](https://3dmatch.cs.princeton.edu/#keypoint-matching-benchmark). We followed [this repo](https://github.com/craigleili/3DLocalMultiViewDesc) to preprocess rgb frames into fused fragments and extract matching keypoints for training. The preprocessed data ready for training can be downloaded [here](https://drive.google.com/file/d/1ME42RjtrNJNz1zSTBrO2NtG89fsOkQLv/view?usp=sharing) (146GB). We also prepared to preprocessed 3DMatch evaluation dataset [here](https://drive.google.com/file/d/14ZGJZHuQLhg87En4C5po6bgTFn4tns4R/view?usp=sharing) (40GB), where local patches around testing keypoints have been precomputed.

<!-- **Pretrained Model**

Pretrained model can be downloaded using this [link](https://drive.google.com/file/d/1vy9FRGWQsuVi4nf--YIqg_8yHFiWWJhh/view?usp=sharing) -->

### ModelNet40 Classification

**Training**

The following lines can be used for the training of each model. In all experiments, `PATH_TO_LOG`, `PATH_TO_MODELNET40`, and `PATH_TO_3DMATCH_TRAIN` are constant and no need to change across experiments. 

```
# E2PN mode
python run_modelnet.py experiment --experiment-id cls_s2(free to choose) --model-dir PATH_TO_LOG -d PATH_TO_MODELNET40 \
model --flag permutation --kanchor 12 --feat-all-anchors --drop_xyz

# EPN mode
python run_modelnet.py experiment --experiment-id cls_so3 --model-dir PATH_TO_LOG -d PATH_TO_MODELNET40 \
train_loss --cls-attention-loss

# KPConv mode
python run_modelnet.py experiment --experiment-id cls_kp --model-dir PATH_TO_LOG -d PATH_TO_MODELNET40 \
model --flag max --kpconv --kanchor 1

# ESCNN mode
python run_modelnet_voxel.py experiment --experiment-id cls_v(free to choose) --model-dir PATH_TO_LOG -d PATH_TO_MODELNET40 \
train --sigma 0.03 model --freq 3 --test_batch_size 24
```

You can set `--debug-mode check_equiv` to make sure the model satisfies the equivariant condition before starting formal training. 

Running ESCNN models (an equivariant baseline based on steerable CNNs and voxel representation) requires installing corresponding packages, see [ESCNN](https://github.com/QUVA-Lab/escnn). You will also need `point_cloud_utils` package. 

**Evaluation**

Use the same command as training, except adding `--run-mode eval` and `-r PATH_TO_CKPT(full path to pth file)`. For example:

```
# E2PN mode
python run_modelnet.py experiment --experiment-id cls_s2 --run-mode eval \
--model-dir PATH_TO_LOG -d PATH_TO_MODELNET40 -r PATH_TO_CKPT \
model --flag permutation --kanchor 12 --feat-all-anchors --drop_xyz
```

You can specify `--shift --jitter --dropout_pt` for random translation, noise, and point dropout as augmentation in training. Set `--train_rot` for different options of training rotational augmentation. Set `--test_aug` to use the training augmentation for testing and `--test_rot` to specify the type of rotations used in testing, or set `--group_test` to test under various conditions all at once. 

### ModelNet40 Shape Alignment
**Training**
```
# E2PN mode
python run_modelnet_rotation.py experiment --experiment-id rot_s2 --model-dir PATH_TO_LOG -d PATH_TO_MODELNET40 \
model --flag permutation --kanchor 12

# EPN mode
python run_modelnet_rotation.py experiment --experiment-id rot_so3 --model-dir PATH_TO_LOG -d PATH_TO_MODELNET40

# KPConv mode
python run_modelnet_rotation.py experiment --experiment-id rot_kp --model-dir PATH_TO_LOG -d PATH_TO_MODELNET40 \
model --kpconv --kanchor 1
```

You can set `--debug-mode check_equiv` to make sure the model satisfies the equivariant condition before starting formal training. 

**Evaluation**

Similarly, add `--run-mode eval` and `-r PATH_TO_CKPT(full path to pth file)`. For example:

```
# E2PN mode
python run_modelnet_rotation.py experiment --experiment-id rot_s2 --run-mode eval \
--model-dir PATH_TO_LOG -d PATH_TO_MODELNET40 -r PATH_TO_CKPT \
model --flag permutation --kanchor 12
```
To obtain reproducible evaluation result, set `-s SOME_INTEGER` as the random seed. 

### 3DMatch Keypoint Matching
**Training**
```
# E2PN mode
python run_3dmatch.py experiment --experiment-id inv_s2 --model-dir PATH_TO_LOG -d PATH_TO_3DMATCH_TRAIN \
train --npt 16 model --kanchor 12

# EPN mode
python run_3dmatch.py experiment --experiment-id inv_so3 --model-dir PATH_TO_LOG -d PATH_TO_3DMATCH_TRAIN \
train --npt 16

# KPConv mode
python run_3dmatch.py experiment --experiment-id inv_kp --model-dir PATH_TO_LOG -d PATH_TO_3DMATCH_TRAIN \
train --npt 16 \
model --kpconv --kanchor 1
```

**Evaluation**

Similarly, add `--run-mode eval` and `-r PATH_TO_CKPT(full path to pth file)`. For example:

```
# E2PN mode
python run_3dmatch.py experiment --experiment-id inv_s2 --run-mode eval \
--model-dir PATH_TO_LOG -d PATH_TO_3DMATCH_EVAL -r PATH_TO_CKPT \
model --kanchor 12
```

## Running on RunPod

These steps use RunPod's official PyTorch template and install the code on the Pod (no custom Docker image). They follow the RunPod documentation, but **this code has not yet been run on CUDA or on RunPod**, so treat the first run as a test and run step 5 before training.

Placeholders used below: `YOUR_GITHUB_USER` (a fork that contains this version of the code), `POD_IP` and `POD_PORT` (shown in the Pod's **Connect** tab under "SSH over exposed TCP"), and `~/.ssh/id_ed25519` (your SSH key).

**1. On your Mac: push the code to a repository you can clone.**

```bash
cd ~/projects/E2PN
git remote add fork https://github.com/YOUR_GITHUB_USER/E2PN.git
git push fork HEAD:main
```

**2. On your Mac: pack the dataset.** ModelNet40 is about 12,000 small files, so send it as one archive.

```bash
cd ~
tar --exclude='.DS_Store' -czf ModelNet40.tar.gz ModelNet40
```

**3. In the RunPod console: create the Pod.**

- Add your SSH public key once under Settings (`cat ~/.ssh/id_ed25519.pub`).
- Deploy a GPU Pod (for example an RTX A6000) from the official **PyTorch** template, such as `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. In that tag `1.0.2` is RunPod's image version, `torch280` is PyTorch 2.8.0 and `cu1281` is CUDA 12.8.1. Any template with PyTorch 2.1 or newer works; pick the newest one offered.
- Keep the volume mounted at `/workspace` (20 GB is plenty: the dataset is about 2 GB and checkpoints are about 11 MB each) and keep TCP port 22 exposed.
- For Weights & Biases logging, add the environment variable `WANDB_API_KEY` with your key.

Only `/workspace` survives a Pod stop. Everything else, including pip-installed packages, is reset.

**4. Copy the dataset to the Pod and connect.**

```bash
# on your Mac
scp -P POD_PORT -i ~/.ssh/id_ed25519 ~/ModelNet40.tar.gz root@POD_IP:/workspace/
ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519
```

```bash
# on the Pod
cd /workspace
tar -xzf ModelNet40.tar.gz && rm ModelNet40.tar.gz
ls ModelNet40 | wc -l          # expect 40 categories
```

**5. On the Pod: install the code and check it.**

```bash
cd /workspace
git clone https://github.com/YOUR_GITHUB_USER/E2PN.git
cd E2PN
pip install --ignore-installed blinker   # the image's blinker comes from apt and pip cannot uninstall it
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest vgtk/tests
```

`torch.cuda.is_available()` must print `True`, and the tests should pass. After a Pod stop and restart, run both `pip install` commands again.

**6. On the Pod: train.** `nohup` keeps the run alive if your SSH session drops.

```bash
cd /workspace/E2PN
nohup python run_modelnet.py experiment --experiment-id cls_s2 --wandb \
  --model-dir /workspace/models -d /workspace/ModelNet40 \
  model --flag permutation --kanchor 12 --feat-all-anchors --drop_xyz \
  > /workspace/train_cls_s2.log 2>&1 &

tail -f /workspace/train_cls_s2.log     # Ctrl-C stops watching, not the training
nvidia-smi                              # GPU utilisation and memory
```

Drop `--wandb` to log only to the file. The run length is set in `run_modelnet.py` (`opt.num_iterations = 80000`); edit that line for a shorter run. To stop a run: `pkill -f run_modelnet.py`.

**7. Evaluate a checkpoint (on the Pod).**

```bash
cd /workspace/E2PN
ls /workspace/models/cls_s2/*/ckpt/
python run_modelnet.py experiment --experiment-id cls_s2 --run-mode eval \
  --model-dir /workspace/models -d /workspace/ModelNet40 \
  -r /workspace/models/cls_s2/MODEL_DIR/ckpt/cls_s2_net_best.pth \
  model --flag permutation --kanchor 12 --feat-all-anchors --drop_xyz
```

**8. On your Mac: download the results, then stop the Pod.**

```bash
scp -r -P POD_PORT -i ~/.ssh/id_ed25519 root@POD_IP:/workspace/models ~/E2PN_models
```

Stop or terminate the Pod in the console when you are done; it is billed while it runs. Stopping keeps `/workspace`; terminating deletes it.

## Citation
If you find our project useful in your research, please consider citing:

```
@inproceedings{zhu2023e2pn,
  title={E2PN: Efficient SE (3)-equivariant point network},
  author={Zhu, Minghan and Ghaffari, Maani and Clark, William A and Peng, Huei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1223--1232},
  year={2023}
}
```
