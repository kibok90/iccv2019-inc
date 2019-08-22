# CIFAR-100, task size 10, seed 0, gpu 0

## Stage 1 models
With and without confidence calibration
```
python main.py --gpu 0 --seed 0 -d cifar100 -e tiny -s res -t 10  5
python main.py --gpu 0 --seed 0 -d cifar100 -s res -t 10  5
```

## Table 1 and figures
#### Without external dataset
Baseline, LwF, DR, E2E, GD (Ours)
```
python main.py --gpu 0 --seed 0 -d cifar100 -s res -t 10 10
python main.py --gpu 0 --seed 0 -d cifar100 -s res -t 10 10 -r L
python main.py --gpu 0 --seed 0 -d cifar100 -s res -t 10 10 -r LC
python main.py --gpu 0 --seed 0 -d cifar100 -s res -t 10 10 -r L   -b dset -f all
python main.py --gpu 0 --seed 0 -d cifar100 -s res -t 10 10 -r PC  -b dw   -f cls
```
#### With external dataset
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
