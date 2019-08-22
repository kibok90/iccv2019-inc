import torch
from torchvision import transforms

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

dataset_stats = {
    'mnist'   : {'mean': (0.13066051707548254,),
                 'std' : (0.30810780244715075,)},
    'fmnist'  : {'mean': (0.28604063146254594,),
                 'std' : (0.35302426207299326,)},
    'cifar10' : {'mean': (0.49139967861519607, 0.48215840839460783, 0.44653091444546567),
                 'std' : (0.2470322324632819, 0.24348512800005573, 0.26158784172796434)},
    'cifar100': {'mean': (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                 'std' : (0.2673342858792409, 0.25643846291708816, 0.2761504713256834)},
    'imagenet': {'mean': (0.4814872465839461, 0.45771731927849263, 0.4082078035692402),
                 'std' : (0.2606993601638989, 0.2536456316098414, 0.2685610203190189)},
                }
# imagenet from 500k training data

def get_transform(dataset='cifar100', phase='test'):
    transform_list = []
    if phase == 'train' and not ('mnist' in dataset):
        transform_list.extend([
            transforms.ColorJitter(brightness=63/255, contrast=0.8),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
                              ])
    transform_list.extend([
            transforms.ToTensor(), \
            transforms.Normalize(dataset_stats[dataset]['mean'], dataset_stats[dataset]['std']),
    ])
    
    return transforms.Compose(transform_list)
