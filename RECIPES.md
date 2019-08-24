# Training recipes
Here are commands to train models on CIFAR-100 (`-d cifar100`) with TinyImages as an external data source (`-e tiny`) when the task size is 10 (`-t 10 10`), using 0-th gpu (`-gpu 0`), as the 0-th trial (`--seed 0`), and save the results in `res/` (`-s res`).

## Base models (0-th stage)
All class-incremental learning models below start from one of these models.

Without confidence calibration
```
python main.py --gpu 0 --seed 0 -d cifar100 -s res -t 10 10
```
With confidence calibration
```
python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10
```

## Table 1 and figures
#### Without external data
Baseline (no distillation)
```
python main.py --gpu 0 --seed 0 -d cifar100 -s res -t 10 10
```
Learning without forgetting [[LwF, ECCV 2016](https://arxiv.org/abs/1606.09282)]
```
python main.py --gpu 0 --seed 0 -d cifar100 -s res -t 10 10 -r L
```
Distillation and retrospection [[DR, ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Saihui_Hou_Progressive_Lifelong_Learning_ECCV_2018_paper.pdf)]
```
python main.py --gpu 0 --seed 0 -d cifar100 -s res -t 10 10 -r LC
```
End-to-end incremental learning [[E2E, ECCV 2018](https://arxiv.org/abs/1807.09536)]
```
python main.py --gpu 0 --seed 0 -d cifar100 -s res -t 10 10 -r L   -b dset -f all
```
Global distillation [GD (Ours)]
```
python main.py --gpu 0 --seed 0 -d cifar100 -s res -t 10 10 -r PC  -b dw   -f cls
```
#### With external data
LwF, DR, E2E, GD (Ours)
```
python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r L
python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r LC
python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r L   -b dset -f all
python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r PCQ -b dw   -f cls
```
## Table 2
```
python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r P   -b dw   -f cls
python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r PC  -b dw   -f cls
python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r Q   -b dw   -f cls
# python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r PCQ -b dw   -f cls
```
## Table 3
```
# python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r P   -b dw   -f cls
python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r PO  -b dw   -f cls --co 0
python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r PO  -b dw   -f cls
python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r PC  -b dw   -f cls --co 0
# python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r PC  -b dw   -f cls
```
## Table 4
```
python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r PCQ -b none
python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r PCQ -b dw
python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r PCQ -b dset -f all
# python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r PCQ -b dw   -f cls
```
## Table 5
```
# python main.py --gpu 0 --seed 0 -d cifar100         -s res -t 10 10 -r PC  -b dw   -f cls
python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r PCQ -b dw   -f cls --ood 1.0
python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r PCQ -b dw   -f cls --ood 0.0
python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r PCQ -b dw   -f cls --oodp
# python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10 10 -r PCQ -b dw   -f cls
```
