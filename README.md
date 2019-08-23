# Introduction
This repository implements [Lee et al. Overcoming Catastrophic Forgetting with Unlabeled Data in the Wild. In ICCV, 2019](https://arxiv.org/abs/1903.12648) in PyTorch.
```
@inproceedings{lee2019overcoming,
  title={Overcoming Catastrophic Forgetting with Unlabeled Data in the Wild},
  author={Lee, Kibok and Lee, Kimin and Shin, Jinwoo and Lee, Honglak},
  booktitle={ICCV},
  year={2019}
}
```
This implementation also includes the state-of-the-art distillation-based methods for class-incremental learning (a.k.a. single-head continual learning):
- Learning without forgetting [[LwF, ECCV 2016](https://arxiv.org/abs/1606.09282)]
- Distillation and retrospection [[DR, ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Saihui_Hou_Progressive_Lifelong_Learning_ECCV_2018_paper.pdf)]
- End-to-end incremental learning [[E2E, ECCV 2018](https://arxiv.org/abs/1807.09536)]

Please see [[training recipes](RECIPES.md)] for replicating them.

# Dependencies
- Python 3.6.8
- NumPy 1.16.2
- PyTorch 0.4.1
- torchvision 0.2.1
- h5py 2.7.1
- tqdm 4.25.0
- tensorboardx 1.4
- SciPy 1.1.0 for `sample_tiny.py`
- matplotlib 2.2.2 for `plotter.py`
- seaborn 0.9.0 for `plotter.py`
- pandas 0.23.0 for `plotter.py`

# Data
You may either generate datasets by yourself or download `h5` files in the following links.
You may not download external data if you don't want to use them.
All data are assumed to be in `data/{dataset}/`. (`{dataset} = cifar100, tiny, imagenet`)

### CIFAR-100 (Training data)
This will be automatically downloaded.

### TinyImages (External data)
- DIY
  - Download [[images (227GB)](http://horatio.cs.nyu.edu/mit/tiny/data/tiny_images.bin)] [[metadata (57GB)](http://horatio.cs.nyu.edu/mit/tiny/data/tiny_metadata.bin)] [[words](http://horatio.cs.nyu.edu/mit/tiny/data/tiny_index.mat)] [[cifar indexes](https://www.cs.toronto.edu/~kriz/cifar_indexes)] and place them in `data/tiny/`.
  - Run `python sample_tiny.py -s {seed}`. `{seed}` corresponds to the stage number in incremental learning.
    - This takes a long time, so running in parallel is recommended.
- Don't DIY
  - Download [[here (20 files; each takes 3GB)](https://drive.google.com/drive/folders/1wk07YzpNXUMfbKmW6BPr3w1RN-KAPe2K)] and place them in `data/tiny/`.
    - Download [[0 (3GB)](https://drive.google.com/open?id=13005jjcI93dn3oe99DNunRyaRQhR50aR)] only, if you use `--ex-static` for training.

### ImageNet (Training and external data)
- DIY
  - Download [[ImageNet ILSVRC 2012 train (154.6GB)](http://image-net.org/download)] and place them in `data/imagenet/ilsvrc2012`.
  - Run the following command. This takes a long time.
    ```
    python image_resizer_imagenet.py -i 'imagenet/ilsvrc2012' -o 'imagenet/ilsvrc2012_resized' -s 32 -a box -r -j 16
    ```
  - Download [[ImageNet 2011 Fall (1.3TB)](http://image-net.org/download)] and place them in `data/imagenet/fall11_whole`.
  - Run the following command. This takes a long time.
    ```
    python image_resizer_imagenet.py -i 'imagenet/fall11_whole' -o 'imagenet/fall11_whole_resized' -s 32 -a box -r -j 16
    ```
  - Run `python sample_imagenet.py -s {seed}`. `{seed}` corresponds to the stage number in incremental learning.
    - Training and test data will be generated at `seed=0`.
    - This takes a long time, so running in parallel is recommended.
- Don't DIY
  - Download [[train (1.5GB)](https://drive.google.com/open?id=1FyaXjtCPg1_33i30--oORtspzFwSAa30)], [[test (0.3GB)](https://drive.google.com/open?id=18vYTxXpVB0lMrMitw3fVm2abGhWW37sN)] and place them in `data/imagenet/`.
  - Download [[here (20 files; each takes 3GB)](https://drive.google.com/drive/folders/194-V0tSOA82mTqFmDzEa83wA6B56Itr2)] and place them in `data/imagenet/`.
    - Download [[0 (3GB)](https://drive.google.com/open?id=1eIw7kVvMM1gzzdq_qx2qaViYF2Qx4mTb)] only, if you use `--ex-static` for training.

# Task splits
- DIY
  - Run `python shuffle_task.py`.
- Don't DIY
  - Task splits are already in `split/`.

# Train and test
- Run `python main.py -h` to see the general usage.
- With `--ex-static`, only 0-th external dataset is used for all stages.
- Please see [[training recipes](RECIPES.md)] for replicating the models compared in our paper.
- Examples on CIFAR-100, task size 10, seed 0, gpu 0:
  - GD (Ours) without external data
    ```
    python main.py --gpu 0 --seed 0 -d cifar100 -s res -t 10 10 -r PC  -b dw -f cls
    ```
  - GD (Ours) with external data
    ```
    python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r PCQ -b dw -f cls
    ```

# Evaluation
- Run `python plotter.py -h` to see the general usage. `bar` and `time` replicate Figure 2(a,b) and (c,d), and the others replicate tables.
- Examples:
  - Replicate CIFAR100 with task size 10 in Table 1
    ```
    python plotter.py -d cifar100 -e tiny -s res -t 10 10 --exp t1
    ```
  - Replicate bar graphs in Figure 2(a,b)
    ```
    python plotter.py -d cifar100 -e tiny -s res --exp bar
    ```
  - Compare arbitrary models you want
    ```
    python plotter.py --exp custom
    ```

# Note
- `image_resizer_imagenet.py` is adapted from [[here](https://github.com/PatrykChrabaszcz/Imagenet32_Scripts/blob/master/image_resizer_imagent.py)]
- `models` is adapted from [[here](https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar)]
- `datasets` is adapted from [[here](https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py)]
