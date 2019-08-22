import os
import sys
import argparse
import time
import random
import copy
import shutil

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn
from torch.utils.data import DataLoader

from tqdm import tqdm
from tensorboardX import SummaryWriter

import datasets, models, utils

dataset_cands = ['cifar100', 'imagenet']
ex_dataset_cands = ['none', 'tiny', 'imagenet']
ref_cands = 'PCQLO'
balance_cands = ['none', 'dset', 'dw']
finetune_cands = ['none', 'cls', 'all']
CUR, PRE, EXT, OOD = 1, 2, 4, 8
ee = 1e-8

parser = argparse.ArgumentParser(description='Commands')

# dir
parser.add_argument('-s', '--save-dir', type=str, default='res', metavar='DIR',
                    help='save directory')
parser.add_argument('--data-dir', type=str, default='data', metavar='DIR',
                    help='data directory')
# dataset
parser.add_argument('-d', '--dataset', type=str, default='cifar100', metavar='DSET',
                    choices=dataset_cands,
                    help='dataset: ' + ' | '.join(dataset_cands) + ' (default: cifar100)')
parser.add_argument('-j', '--workers', type=int, default=2, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-t', '--task-size', type=int, nargs='+', default=[10, 10], metavar='N+',
                    help='number of initial classes and incremental classes (default: 10 10)')
# coreset
parser.add_argument('-c', '--coreset', type=int, default=2000, metavar='N',
                    help='the size of coreset; if 0, memory-less training (default: 2000)')
# ext
parser.add_argument('-e', '--ex-dataset', type=str, default='none', metavar='DSET',
                    choices=ex_dataset_cands,
                    help='external dataset: ' + ' | '.join(ex_dataset_cands) + ' (default: none)')
parser.add_argument('--ex-aux', type=str, default='', metavar='STR',
                    help='controlled external dataset (default: "")')
parser.add_argument('--ex-static', action='store_true', default=False,
                    help='static external dataset (default: False)')
# training
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train including fine-tuning (default: 200)')
parser.add_argument('--lr', type=float, default=0.1, metavar='R',
                    help='learning rate (default: 0.1)')
parser.add_argument('--lrd', type=float, default=0.1, metavar='R',
                    help='learning rate decay (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='R',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='R',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--schedule', type=int, nargs='+', default=[120, 160, 180], metavar='N+',
                    help='when to decay SGD learning rate without fine-tuning (default: 120 160 180)')
# distillation
parser.add_argument('-r', '--ref', type=str, default='', metavar='STR',
                    help='reference models; P: prev, L: local prev, C: cur, O: one-hot cur, Q: ensemble (default: "")')
parser.add_argument('--kdr', '--kd-ratio', type=float, default=1., metavar='R',
                    help='knowledge distillation loss ratio (default: 1.)')
parser.add_argument('--T', '--temperature', type=float, default=2., metavar='R',
                    help='temperature for knowledge distillation (default: 2.)')
parser.add_argument('--qT', '--q-temperature', type=float, default=1., metavar='R',
                    help='temperature for global distillation from Q (default: 1.)')
# balance/confidence/sample
parser.add_argument('-b', '--balance', type=str, default='none', metavar='STR',
                    choices=balance_cands,
                    help='balancing data: ' + ' | '.join(balance_cands) + ' (default: none)')
parser.add_argument('--co', '--conf-out', type=float, default=1., metavar='R',
                    help='out-of-distribution confidence loss ratio (default: 0.)')
parser.add_argument('--exr', '--ex-ratio', type=float, default=1., metavar='R',
                    help='external data ratio (default: 1.)')
parser.add_argument('--oodp', '--ood-pred', action='store_true', default=False,
                    help='sample OOD data based on prediction (default: False)')
parser.add_argument('--ood', '--ood-ratio', type=float, default=0.7, metavar='PROB',
                    help='ratio of OOD data when sampling (default: 0.7)')
parser.add_argument('--ss', '--sample-scale', type=float, default=1., metavar='R',
                    help='scale the number of external dataset to be sampled (default: 1.)')
# fine-tune
parser.add_argument('-f', '--finetune', type=str, default='none', metavar='STR',
                    choices=finetune_cands,
                    help='do fine-tuning ' + ' | '.join(finetune_cands) + ' (default: none)')
parser.add_argument('--fepochs', type=int, default=20, metavar='N',
                    help='number of epochs to fine-tune (default: 20)')
parser.add_argument('--fschedule', type=int, nargs='+', default=[10, 15], metavar='N+',
                    help='when to decay SGD learning rate during fine-tuning (default: 10 15)')
parser.add_argument('--mschedule', type=int, nargs='+', default=[120, 160, 170], metavar='N+',
                    help='when to decay SGD learning rate with fine-tuning (default: 120 160 170)')
# reuse
parser.add_argument('--restart', action='store_true', default=False,
                    help='ignore existing results (default: False)')
parser.add_argument('--rtm', '--retrain-model', action='store_true', default=False,
                    help='retrain the model from random initialization (default: False)')
parser.add_argument('--rtc', action='store_true', default=False,
                    help='retrain the current teacher from random initialization (default: False)')
# arch
parser.add_argument('-a', '--arch', type=str, default='wrn', metavar='ARCH',
                    help='CNN architecture (default: wrn)')
parser.add_argument('--depth', type=int, default=16, metavar='N',
                    help='Model depth')
parser.add_argument('--wf', '--widen-factor', type=int, default=2, metavar='N',
                    help='Widen factor for WRN')
parser.add_argument('--drop', '--dropout', default=0.3, type=float, metavar='PROB',
                    help='Dropout ratio')
# device
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 0)')
parser.add_argument('--gpu', default='0', type=str, metavar='N,',
                    help='argument for CUDA_VISIBLE_DEVICES (default: 0)')

# parse
args = parser.parse_args()

data_dir = args.data_dir
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
save_dir = args.save_dir
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# deterministic random numbers
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def adjust_learning_rate(optimizer, lrd, epoch, schedule):
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            print('lr decay from {} to {}'.format(param_group['lr'], param_group['lr'] * lrd))
            param_group['lr'] *= lrd

def write_to_tb(writer, res, phase='train', epoch=0):
    for key in res:
        if isinstance(res[key], dict):
            writer.add_scalars('{}/{}'.format(phase, key), res[key], epoch)
        else:
            writer.add_scalar ('{}/{}'.format(phase, key), res[key], epoch)

def get_performance(output, target):
    acc = (output.max(1)[1] == target).to(torch.float).mean()
    return acc

def concat_target(target, aug, cur, ood, task_info):
    task, seen, prev = task_info
    target_full = target.new_zeros(target.size(0), max(seen)+1)
    # scale based on non-target probs
    l, r = len(prev), len(seen)
    inc = len(task) # r-l
    if True: # target in prev
        cur = ood = target.new_zeros(target.size(0), dtype=torch.uint8)
    elif False: # target in prev or ood
        cur = target.new_zeros(target.size(0), dtype=torch.uint8)
        if ood is None: ood = target.new_zeros(target.size(0), dtype=torch.uint8)
    else: # target in prev or cur or ood
        if cur is None: cur = target.new_zeros(target.size(0), dtype=torch.uint8)
        if ood is None: ood = target.new_zeros(target.size(0), dtype=torch.uint8)
    old = ~(cur | ood)
    if aug is None: aug = target.new_ones(target.size(0), inc) / inc
    # [pmax, p=(1-pmax-eps), c=eps] or [pmax, p=(1-eps), c=(eps-pmax)]
    if old.any():
        target_old = target.new_zeros(old.sum(), max(seen)+1)
        target_max, target_amax = target[old].max(dim=1, keepdim=True)
        eps = (1.-target_max) * inc / (r-1)
        target_old[:,prev] = target[old] * (l-1)/(r-1)
        target_old[:,task] = aug   [old] * eps
        target_old[range(target_old.size(0)),prev[target_amax[:,0]]] = target_max[:,0]
        target_full[old]   = target_old
    if cur.any():
        target_cur = target.new_zeros(cur.sum(), max(seen)+1)
        aug_max, aug_amax = aug[cur].max(dim=1, keepdim=True)
        eps = (aug_max*l + inc-1) / (r-1)
        target_cur[:,prev] = target[cur] * (1.-eps)
        target_cur[:,task] = aug   [cur] * (inc-1)/(r-1)
        target_cur[range(target_cur.size(0)),task[aug_amax[:,0]]] = aug_max[:,0]
        target_full[cur]   = target_cur
    if ood.any():
        target_ood = target.new_zeros(ood.sum(), max(seen)+1)
        target_ood[:,prev] = target[ood] * l/r
        target_ood[:,task] = aug   [ood] * (r-l)/r
        target_full[ood]   = target_ood
    target_pad = target_full[:,seen]
    return target_pad

def train0(train_loader, model, criterion, dw, optimizer, task_info, t, epoch):

    task, class_map = task_info
    task = torch.tensor(task).to(device)
    class_map = torch.tensor(class_map).to(device)

    model.train()
    loss_ = utils.AverageMeter()
    acc_  = utils.AverageMeter()
    pbar  = tqdm(train_loader, desc='tr {:d}'.format(epoch), ascii=True, ncols=80)
    for data, target, src in pbar:

        data, target, src = data.to(device), target.to(device), src.to(device)

        optimizer.zero_grad()
        output, feature = model(data)

        loc_cur = (src & CUR) > 0
        loc_conf = ~loc_cur

        loss = output.new_zeros(1)

        # classification
        if loc_cur.any():
            output_cls = output[loc_cur][:,task]
            target_cls = class_map[target[loc_cur]]
            dw_cls     = dw['local'][target_cls]
            loss += (criterion(output_cls, target_cls) * dw_cls).mean()

        # confidence
        if (args.co > 0.) and loc_conf.any():
            output_conf = output[loc_conf][:,task]
            loss -= (output_conf.log_softmax(dim=1).mean() + math.log(len(task))) * args.co

        try:
            loss.backward()
            optimizer.step()
        except RuntimeError:
            with open('{}/err.txt'.format(save_dir), 'a') as f:
                f.write('error at train0, t={:d}, e={:d}\n'.format(t, epoch))

        if loc_cur.any():
            acc = get_performance(output_cls, target_cls)
        else:
            print('Warning: no current task data in this batch')
            acc = torch.zeros(1)

        loss_.update(loss.item(), data.size(0))
        acc_.update(acc.item(), data.size(0))
        pbar.set_postfix(acc='{:5.2f}'.format(acc_.avg*100.), loss='{:.4f}'.format(loss_.avg))

    res = {'loss': {'val': loss_.avg}, 'acc' : {'val': acc_.avg*100.}}
    return res

def train(train_loader, model_dict, criterion, dw, optimizer, p_knows_t, loss_for_t, task_info, t, epoch):

    tasks, class_maps, prev, prev_map, seen, seen_map = task_info
    tasks = [torch.tensor(task).to(device) for task in tasks]
    class_maps = [torch.tensor(class_map).to(device) for class_map in class_maps]
    prev, seen = torch.tensor(prev).to(device), torch.tensor(seen).to(device)
    prev_map, seen_map = torch.tensor(prev_map).to(device), torch.tensor(seen_map).to(device)
    if 'Q' in args.ref:
        task_info_pad = (tasks[t], seen, prev)

    model_dict['m'].train()
    if model_dict['p'] is not None: model_dict['p'].eval()
    if model_dict['c'] is not None: model_dict['c'].eval()
    if p_knows_t:
        tt = t+1
        prev = seen
        prev_map = seen_map
    else:
        tt = t
    num_local = len(seen) if loss_for_t else len(prev)

    loss_ = utils.AverageMeter()
    acc_  = utils.AverageMeter()
    pbar  = tqdm(train_loader, desc='tr {:d}'.format(epoch), ascii=True, ncols=80)
    for data, target, src in pbar:

        data, target, src = data.to(device), target.to(device), src.to(device)

        optimizer.zero_grad()
        output, feature = model_dict['m'](data)
        if model_dict['p'] is None: output_p = None
        else:                       output_p, _ = model_dict['p'](data)
        if model_dict['c'] is None: output_c = None
        else:                       output_c, _ = model_dict['c'](data)

        if p_knows_t:
            loc_prev = (src & (PRE | CUR)) > 0
            loc_cur  = torch.zeros_like(src, dtype=torch.uint8)
            loc_seen = loc_prev
        else:
            loc_prev = (src & PRE) > 0
            loc_cur  = (src & CUR) > 0
            loc_seen = loc_prev | loc_cur
        loc_ood  = (src & OOD) > 0
        loc_ext  = (src & (EXT | OOD)) > 0

        loss = output.new_zeros(1)

        # classification
        if loc_seen.any():
            output_cls = output[loc_seen][:,seen]
            target_cls = seen_map[target[loc_seen]]
            dw_cls     = dw['seen'][target_cls]
            loss += (criterion['cls'](output_cls, target_cls) * dw_cls).sum()

        # distillation from Q (ensemble of P and C)
        if ('Q' in args.ref) and loc_ext.any() and (not p_knows_t):
            output_q  = (output  [loc_ext][:,seen] / args.qT).log_softmax(dim=1)
            starget_q = (output_p[loc_ext][:,prev] / args.qT).softmax(dim=1)
            target_q  = seen_map[target[loc_ext]]
            dw_q      = dw['seen'][target_q][:,None]
            if output_c is None: starget_t = None
            else:                starget_t = (output_c[loc_ext][:,tasks[t]] / args.qT).softmax(dim=1)
            starget_q = concat_target(starget_q, starget_t, loc_cur[loc_ext], loc_ood[loc_ext], task_info_pad)
            loss += (criterion['dst'](output_q, starget_q) * dw_q).sum(dim=1).sum() * (args.qT**2) * args.kdr

            # normalize global losses together
            loss /= (loc_seen.to(torch.float).sum() + loc_ext.to(torch.float).sum())
        else:
            loss /= loc_seen.to(torch.float).sum()

        # distillation from P (previous model)
        if 'P' in args.ref:
            output_pgd  = (output  [:,prev] / args.T).log_softmax(dim=1)
            starget_pgd = (output_p[:,prev] / args.T).softmax(dim=1)
            target_pgd  = prev_map[target]
            dw_pgd      = dw['prev'][target_pgd][:,None]
            bloss_pgd   = len(prev) / num_local
            loss += (criterion['dst'](output_pgd, starget_pgd) * dw_pgd).sum(dim=1).mean() * (args.T**2) * args.kdr * bloss_pgd

        # distillation from C (teacher for the current task)
        if ('C' in args.ref) and (not p_knows_t) and (output_c is not None):
            output_cdst  = (output  [:,tasks[t]] / args.T).log_softmax(dim=1)
            starget_cdst = (output_c[:,tasks[t]] / args.T).softmax(dim=1)
            target_cdst  = class_maps[t][target]
            dw_cdst      = dw['local'][t][target_cdst][:,None]
            bloss_cdst   = len(tasks[t]) / num_local
            loss += (criterion['dst'](output_cdst, starget_cdst) * dw_cdst).sum(dim=1).mean() * (args.T**2) * args.kdr * bloss_cdst

        # classification at t (one-hot teacher for the current task)
        if 'O' in args.ref:
            if loc_cur.any():
                output_ccls = output[loc_cur][:,tasks[t]]
                target_ccls = class_maps[t][target[loc_cur]]
                dw_ccls     = dw['local'][t][target_ccls]
                bloss_ccls  = len(tasks[t]) / num_local
                loss += (criterion['cls'](output_ccls, target_ccls) * dw_ccls).mean() * bloss_ccls

            # confidence
            loc_conf = ~loc_cur
            if (args.co > 0.) and loc_conf.any():
                output_conf = output[loc_conf][:,tasks[t]]
                bloss_conf  = len(tasks[t]) / num_local
                loss -= (output_conf.log_softmax(dim=1).mean() + math.log(len(tasks[t]))) * args.co * args.kdr * bloss_conf

        # local distillation from P (previous model)
        if 'L' in args.ref:
            for s, task in enumerate(tasks[:tt]):
                output_pld  = (output  [:,task] / args.T).log_softmax(dim=1)
                starget_pld = (output_p[:,task] / args.T).softmax(dim=1)
                target_pld  = class_maps[s][target]
                dw_pld      = dw['local'][s][target_pld][:,None]
                bloss_pld   = len(task) / num_local
                loss += (criterion['dst'](output_pld, starget_pld) * dw_pld).sum(dim=1).mean() * (args.T**2) * args.kdr * bloss_pld

        try:
            loss.backward()
            optimizer.step()
        except RuntimeError:
            with open('{}/err.txt'.format(save_dir), 'a') as f:
                f.write('error at train t={:d}, e={:d}\n'.format(t, epoch))

        if loc_seen.any():
            acc = get_performance(output_cls, target_cls)
        else:
            print('Warning: no labeled data in this batch')
            acc = torch.zeros(1)

        loss_.update(loss.item(), data.size(0))
        acc_.update(acc.item(), data.size(0))
        pbar.set_postfix(acc='{:5.2f}'.format(acc_.avg*100.), loss='{:.4f}'.format(loss_.avg))

    res = {'loss': {'val': loss_.avg}, 'acc' : {'val': acc_.avg*100.}}
    return res

def test0(test_loader, model, criterion, task_info, t, epoch):

    task, class_map = task_info
    task = torch.tensor(task).to(device)
    class_map = torch.tensor(class_map).to(device)

    model.eval()

    loss_ = utils.AverageMeter()
    acc_  = utils.AverageMeter()
    pbar  = tqdm(test_loader, desc='te {:d}'.format(epoch), ascii=True, ncols=80)
    for data, target, _ in pbar:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output, _ = model(data)
            loss = criterion(output[:,task], class_map[target]).mean()
        acc = get_performance(output[:,task], class_map[target])

        loss_.update(loss.item(), data.size(0))
        acc_.update(acc.item(), data.size(0))
        pbar.set_postfix(acc='{:5.2f}'.format(acc_.avg*100.), loss='{:.4f}'.format(loss_.avg))

    res = {'loss': {'val': loss_.avg}, 'acc' : {'mean': acc_.avg*100.}}
    for s in range(1):
        res['acc'].update({'{}'.format(s): acc_.avg*100.})
    return res

def test(test_loader, model, criterion, task_info, t, epoch):

    tasks, class_maps, _, _, seen, seen_map = task_info
    tasks = [torch.tensor(task).to(device) for task in tasks]
    class_maps, seen_map = [torch.tensor(class_map).to(device) for class_map in class_maps], torch.tensor(seen_map).to(device)

    model.eval()

    loss_  = utils.AverageMeter()
    acc_   = utils.AverageMeter()
    acc_t_ = [utils.AverageMeter() for _ in range(t+1)]
    pbar   = tqdm(test_loader, desc='te {:d}'.format(epoch), ascii=True, ncols=80)
    for data, target, _ in pbar:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output, feature = model(data)
            loss = criterion(output[:,seen], seen_map[target]).mean()
        acc = get_performance(output[:,seen], seen_map[target])

        loss_.update(loss.item(), data.size(0))
        acc_.update(acc.item(), data.size(0))
        pbar.set_postfix(acc='{:5.2f}'.format(acc_.avg*100.), loss='{:.4f}'.format(loss_.avg))

        for s, task in enumerate(tasks[:t+1]):
            loc = torch.zeros_like(target, dtype=torch.uint8)
            for k in task:
                loc |= (target == k)
            num_data_s = loc.to(torch.long).sum()
            if num_data_s > 0:
                acc = get_performance(output[loc][:,seen], seen_map[target[loc]])
                acc_t_[s].update(acc.item(), loc.to(torch.long).sum().item())

    if epoch+1 == args.epochs:
        for s in range(t+1):
            print('{:d}:{:5.2f}, '.format(s, acc_t_[s].avg*100.), end='')
        print()

    res = {'loss': {'val': loss_.avg}, 'acc' : {'mean': acc_.avg*100.}}
    for s in range(t+1):
        res['acc'].update({'{}'.format(s): acc_t_[s].avg*100.})
    return res

def sample_data(num_data, dataset, model, classes, kwargs={}):
    print('before sampling data: dataset has {} samples'.format(len(dataset)))
    if args.ss != 1.:
        num_data = int(num_data * args.ss)
    if len(classes) == 0: # no in-distribution
        osmp = num_data
        ismp = 0
        locs = np.random.choice(len(dataset), num_data, replace=False)
        dataset.ood = dataset.data[locs]
        dataset.data = np.zeros(np.concatenate([[0], dataset.data.shape[1:]]), dtype=dataset.data.dtype)
        dataset.targets = np.zeros(0, dtype=dataset.targets.dtype)
    else:
        # outputs of the model
        osmp = round(num_data * args.ood)
        ismp = num_data - osmp
        transform = dataset.transform
        dataset.transform = utils.get_transform(dataset=args.dataset, phase='test')
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)
        model.eval()
        pbar = tqdm(data_loader, desc='sample', ascii=True, ncols=80)
        outputs = []
        for data, _, _ in pbar:
            data = data.to(device)
            with torch.no_grad():
                output, _ = model(data)
            outputs.append(output[:,classes])
        outputs = torch.cat(outputs, dim=0)

        # in-distribution
        locs = torch.zeros(outputs.size(0)).to(torch.uint8)
        if ismp > 0:
            for k in range(len(classes)):
                locs[outputs[:,k].topk(ismp // len(classes))[1]] = 1

        # out-of-distribution
        locs_ood = torch.zeros(outputs.size(0)).to(torch.uint8)
        if osmp > 0:
            if args.oodp:
                locs_ood[outputs.log_softmax(dim=1).mean(dim=1).topk(osmp)[1]] = 1
            else:
                locs_ood[torch.tensor(np.random.choice(locs_ood.size(0), osmp, replace=False))] = 1
            locs_ood[locs] = 0

        locs = locs.to('cpu').numpy().astype(bool)
        locs_ood = locs_ood.to('cpu').numpy().astype(bool)
        dataset.ood = dataset.data[locs_ood]
        dataset.data = dataset.data[locs]
        dataset.targets = dataset.targets[locs]

        dataset.transform = transform
    print('after  sampling data: dataset has {}/{} ext'.format(len(dataset), ismp))
    if hasattr(dataset, 'ood') and (dataset.ood is not None):
        print('after  sampling data: dataset has {}/{} ood'.format(len(dataset.ood), osmp))

def get_data_weight(dataset, tasks, prev, seen, t, phase, apply_dw=False, p_knows_t=False):

    if p_knows_t:
        prev = seen
    dwex = (args.finetune == 'none') and apply_dw

    if phase == 'init':
        stats_seen, stats_prev, stats_local = \
            dataset.get_stats(prev, seen, d_only_in_t=True, p_knows_t=False,dwex=False, dwood=False)

        dw_seen = dw_prev = None

        dw_local = np.ones(len(tasks[t])+2, dtype=np.float32)
        dw_local[-2:] = args.exr
        if apply_dw:
            stats_ls = stats_local[t]
            dw_local[:len(tasks[t])] = stats_ls.sum() / (stats_ls[stats_ls > 0] * len(tasks[t]))
            # if stats_ls.sum() > 0:
            #     dw_local[-1] = dw_local[-1] / (1. / dw_local[:len(tasks[t])]).mean()
        dw_local = torch.tensor(dw_local).to(device)

    else:
        stats_seen, stats_prev, stats_local = \
            dataset.get_stats(prev, seen, d_only_in_t=('O' in args.ref), p_knows_t=p_knows_t, dwex=dwex, dwood=dwex)

        dw_seen = np.ones(len(seen)+2, dtype=np.float32)
        dw_seen[-2:] = args.exr
        if apply_dw:
            dw_seen[:len(seen)][stats_seen > 0] = stats_seen.sum() / (stats_seen[stats_seen > 0] * len(seen))
            # if stats_seen.sum() > 0:
            #     dw_seen[-1] = dw_seen[-1] / (1. / dw_seen[:len(seen)][stats_seen > 0]).mean()
        dw_seen = torch.tensor(dw_seen).to(device)

        dw_prev = np.ones(len(prev)+2, dtype=np.float32)
        dw_prev[-2:] = args.exr
        if apply_dw:
            dw_prev[:len(prev)][stats_prev > 0] = stats_prev.sum() / (stats_prev[stats_prev > 0] * len(prev))
            # if stats_prev.sum() > 0:
            #     dw_prev[-1] = dw_prev[-1] / (1. / dw_prev[:len(prev)][stats_prev > 0]).mean()
        dw_prev = torch.tensor(dw_prev).to(device)

        dw_local = [np.ones([len(task)+2], dtype=np.float32) for task in tasks[:t+1]]
        for s in range(t+1):
            dw_local[s][-2:] = args.exr
            if apply_dw:
                stats_ls = stats_local[s]
                dw_local[s][:len(tasks[s])][stats_ls > 0] = stats_ls.sum() / (stats_ls[stats_ls > 0] * len(tasks[s]))
                # if stats_ls.sum() > 0:
                #     dw_local[s][-1] = dw_local[s][-1] / (1. / dw_local[s][:len(tasks[s])][stats_ls > 0]).mean()
        dw_local = [torch.tensor(dw_ls).to(device) for dw_ls in dw_local]

    print(len(dataset))
    print(stats_seen)
    print(stats_prev)
    print(stats_local)
    print(dw_seen)
    print(dw_prev)
    print(dw_local)

    return dw_seen, dw_prev, dw_local

if __name__ == '__main__':

    # sanity check
    if len(args.ref) == 0:
        args.T = args.kdr = 0.
    if 'Q' not in args.ref:
        args.qT = 0.

    print(args)

    # tasks
    num_classes = 100
    class_order = np.load('split/class_order_{}.npy'.format(num_classes))[args.seed].tolist()

    tasks = []
    class_maps = []
    p = 0
    while p < num_classes:
        inc = args.task_size[1] if p > 0 else args.task_size[0]
        tasks.append(class_order[p:p+inc])
        class_map = np.full(num_classes, -1)
        for i, j in enumerate(tasks[-1]): class_map[j] = i
        class_maps.append(class_map)
        p += inc
    num_tasks = len(tasks)

    # model
    base_str = args.dataset
    base_str += '_{seed:d}'.format(seed=args.seed)
    if args.arch == 'resnet':
        model = models.resnet(depth=args.depth, num_classes=num_classes).to(device)
        # base_str += '_{arch}-{depth}_wd={wd:.0e}' \
                   # .format(arch=args.arch, depth=args.depth, wd=args.wd)
    elif args.arch == 'wrn':
        model = models.wrn(depth=args.depth, num_classes=num_classes, widen_factor=args.wf, dropRate=args.drop).to(device)
        # base_str += '_{arch}_{depth}_{wf}_{drop:.1f}_wd={wd:.0e}' \
                   # .format(arch=args.arch, depth=args.depth, wf=args.wf, drop=args.drop, wd=args.wd)
    else:
        raise ValueError('"{}" is not supported.'.format(args.arch))

    if len(args.gpu) > 1:
        model = nn.DataParallel(model)

    # save str
    new_str = '{base_str}'.format(base_str=base_str)
    e_str = 'e={ex_dataset}_co={co:.1f}'.format(ex_dataset=args.ex_dataset, co=args.co)
    if len(args.ex_aux) > 0:
        e_str += '_{ex_aux}'.format(ex_aux=args.ex_aux)
    if args.ex_dataset != 'none':
        new_str += '_{e_str}'.format(e_str=e_str)
        if args.co > 0:
            base_str += '_{e_str}'.format(e_str=e_str)
    elif ('C' in args.ref) or ('O' in args.ref):
        new_str += '_co={co:.1f}'.format(co=args.co)

    base_str += '_t={bt}'.format(bt=args.task_size[0])
    new_str  += '_t={bt}_{it}'.format(bt=args.task_size[0], it=args.task_size[1])

    if args.coreset > 0:
        new_str += '_cs={cs:d}'.format(cs=args.coreset)
    if len(args.ref) > 0:
        new_str += '_ref={ref}_kdr={kdr:.1f}'.format(ref=args.ref, kdr=args.kdr)
        if ('P' in args.ref) or ('C' in args.ref) or ('L' in args.ref):
            new_str += '_T={T:.1f}'.format(T=args.T)
        if 'Q' in args.ref:
            new_str += '_qT={qT:.1f}'.format(qT=args.qT)
    new_str += '_bd={bd}'.format(bd=args.balance)
    if args.finetune != 'none':
        new_str += '_ft={finetune}'.format(finetune=args.finetune)
    if args.ex_dataset != 'none':
        new_str += '_exr={exr:.1f}'.format(exr=args.exr)
        new_str += '_{oodp}={ood:.1f}'.format(oodp='oodp' if args.oodp else 'ood', ood=args.ood)
        if args.ss != 1.:
            new_str += '_ss={ss:.1f}'.format(ss=args.ss)
    if args.rtm:
        new_str += '_rtm'
    if args.rtc:
        new_str += '_rtc'

    print(base_str)
    print(new_str)

    # data loader
    Dataset = datasets.iCIFAR100 if args.dataset == 'cifar100' else datasets.H5Dataset

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
    root = os.path.join(data_dir, args.dataset)
    if args.ex_dataset != 'none':
        eroot = os.path.join(data_dir, args.ex_dataset)
        etype = 'out'
        preload_ex = len(args.ex_aux) > 0
        if preload_ex:
            etype += ('_' + args.ex_aux)
        num_ex = 1000000
    else:
        eroot = ''
        etype = 'none'
        preload_ex = False
        num_ex = 0

    train_transform = utils.get_transform(dataset=args.dataset, phase='train')
    test_transform  = utils.get_transform(dataset=args.dataset, phase='test')
    train_dataset = Dataset(root, args.dataset, train=True, download=True, transform=train_transform,
                            slabels=[CUR, PRE, EXT, OOD], tasks=tasks, exr=args.exr, seed=args.seed)
    test_dataset  = Dataset(root, args.dataset, train=False, download=False, transform=test_transform,
                            tasks=tasks, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader  = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, **kwargs)

    # external data loader
    if args.ex_dataset != 'none':
        ex_dataset = datasets.H5Dataset(eroot, dataset=args.ex_dataset, train=False, download=False,
                                        transform=test_transform, seed=0, etype='all' if preload_ex else etype,
                                        num_ex=num_ex, load=preload_ex)
    else:
        ex_dataset = None

    print('training dataset size: {:d}'.format(len(train_dataset)))

    writer = None
    accs = []
    accs_cls = []
    for t in range(num_tasks):

        # tasks for the current stage
        prev = sorted(set([k for task in tasks[:t] for k in task]))
        seen = sorted(set([k for task in tasks[:t+1] for k in task]))
        prev_map = np.full(num_classes, -1)
        seen_map = np.full(num_classes, -1)
        for i, j in enumerate(prev): prev_map[j] = i
        for i, j in enumerate(seen): seen_map[j] = i

        # random seed
        random.seed(args.seed*100+t)
        np.random.seed(args.seed*100+t)
        torch.manual_seed(args.seed*100+t)
        torch.cuda.manual_seed(args.seed*100+t)

        # flags
        use_ex = (args.ex_dataset != 'none')
        do_init = ((t == 0) or ('C' in args.ref))
        use_ex_in_init = use_ex and do_init and (args.co > 0.)
        use_coreset_in_init = (args.coreset > 0) and (t > 0) and do_init and (args.co > 0.)

        # load dataset
        train_dataset.load_dataset(prev, t, train=True)
        test_dataset.load_dataset(prev, t, train=False)

        # save str
        save_str = base_str if t == 0 else new_str

        # find recent model and log
        model_path = '{}/model_{}_{:d}.pt'.format(save_dir, save_str, t)
        if os.path.isfile(model_path) and ((t == 0) or (not args.restart)):
            model_state_dict = torch.load(model_path)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(model_state_dict)
            else:
                model.load_state_dict(model_state_dict)
            print('{:d}-th results loaded'.format(t))

            if (args.coreset > 0) and (t > 0):
                # append coreset
                train_dataset.append_coreset(only=False)

                # update and append coreset if fine-tuning takes a balanced dataset
                if (args.finetune != 'none') and (args.balance == 'dset'):
                    coreset_size = args.coreset + (1 + (args.coreset-1)//len(prev)) * (len(seen)-len(prev))
                    train_dataset.update_coreset(coreset_size, seen)
                    train_dataset.append_coreset(only=True)

            # update coreset
            if args.coreset > 0:
                train_dataset.update_coreset(args.coreset, seen)

            continue

        elif (t == 1) and args.restart:
            p_str = '{}/p_{}.txt'.format(save_dir, save_str)
            if os.path.isfile(p_str):
                os.remove(p_str)
            q_str = '{}/q_{}.txt'.format(save_dir, save_str)
            if os.path.isfile(q_str):
                os.remove(q_str)

        # m: new model, p: previous model, c: teacher for the current task
        model_dict = {'m': model, 'p': None, 'c': None}

        # logger
        writer_path = '{}/{}'.format(save_dir, save_str)
        log_exist = os.path.isdir(writer_path)
        if (t == 1) and (not log_exist):
            base_writer_path = '{}/{}'.format(save_dir, base_str)
            shutil.copytree(base_writer_path, writer_path)
        if (t == 1) or (writer is None):
            writer = SummaryWriter(writer_path)
            if log_exist:
                writer.file_writer.reopen()
        json_path = '{}/log_{}_{:d}.json'.format(save_dir, save_str, t)

        # training
        start_time = time.time()
        phases = []
        if do_init:
            phases.append('init')
        if t > 0:
            phases.append('main')
            if args.finetune != 'none':
                phases.append('finetune')
        for phase in phases:

            # whether P knows the current task
            p_knows_t = False

            # manage models
            if phase == 'init':
                # initialize C
                model = model_dict['c'] = copy.deepcopy(model_dict['m'])
                # reinitizalize C if specified
                if (t > 0) and args.rtc:
                    models.utils.init_module(model_dict['c'])
            else:
                model = model_dict['m']
            # transfer M to P unless fine-tuning phase with C
            if (phase != 'finetune') or ('C' not in args.ref):
                model_dict['p'] = copy.deepcopy(model_dict['m'])
            # remove C if fine-tuning phase without C
            if (phase == 'finetune') and ('C' not in args.ref):
                model_dict['c'] = None
                p_knows_t = True
            # reinitizalize M if specified
            if (phase == 'main') and args.rtm:
                models.utils.init_module(model)

            model_classifier = model.module.classifier if isinstance(model, nn.DataParallel) else model.classifier

            # update requires_grad
            if phase == 'init': model_classifier.set_trainable(tasks[t])
            else:               model_classifier.set_trainable(seen)

            # manage dataset
            if ((phase == 'init') and use_ex_in_init) or \
               ((phase == 'main') and use_ex and (not use_ex_in_init)):
                # load and sample external dataset
                ex_dataset.load_from_file(seed=(0 if args.ex_static else t), etype=etype, load_index=preload_ex)
                print('{:d}-th external dataset size: {:d}'.format(t, len(ex_dataset)))
                num_data = len(train_dataset) + len(train_dataset.coreset[0])
                sample_data(num_data, ex_dataset, model_dict['p'], prev, kwargs)

                # append external dataset
                train_dataset.append_ex(ex_dataset, keep_label=False)

            if ((phase == 'init') and use_coreset_in_init) or \
               ((phase == 'main') and (args.coreset > 0) and (not use_coreset_in_init)):
                train_dataset.append_coreset(only=False)

            if phase == 'finetune':
                # remove external dataset if not specified
                if use_ex and (args.balance == 'dset'):
                    train_dataset.remove_ex()

            # balance
            balance = 'none' if (args.finetune != 'none') and (phase != 'finetune') else args.balance

            # balance dataset
            if (args.coreset > 0) and (balance == 'dset'):
                coreset_size = args.coreset + (1 + (args.coreset-1)//len(prev)) * (len(seen)-len(prev))
                train_dataset.update_coreset(coreset_size, seen)
                train_dataset.append_coreset(only=True)

            # balance data weight
            dw = get_data_weight(train_dataset, tasks, prev, seen, t, phase,
                                 apply_dw=(balance == 'dw'), p_knows_t=p_knows_t)
            dw = {'seen': dw[0], 'prev': dw[1], 'local': dw[2]}

            loss_for_t = ((phase == 'main') and (('C' in args.ref) or ('O' in args.ref))) or (phase == 'finetune')

            # learning rate
            init_lr = args.lr
            if phase == 'finetune':
                init_lr *= args.lrd

            # optimizer
            print('{}: lr {}'.format(phase, init_lr))
            if (phase == 'finetune') and (args.finetune == 'cls'):
                params = filter(lambda p: p.requires_grad, model_classifier.parameters())
            else:
                params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = optim.SGD(params, lr=init_lr, momentum=args.momentum, weight_decay=args.wd)

            # criterion
            criterion = {}
            criterion.update({'cls': nn.CrossEntropyLoss(reduction='none', ignore_index=-1).to(device)})
            criterion.update({'dst': nn.KLDivLoss(reduction='none').to(device)})

            result_to_tb = (t == 0) or (phase != 'init')

            if phase == 'init':
                start_epoch = 0
                end_epoch   = args.epochs
                schedule    = args.schedule
            elif phase == 'main':
                start_epoch = 0
                if 'finetune' in phases:
                    end_epoch   = args.epochs - args.fepochs
                    schedule    = args.mschedule
                else:
                    end_epoch   = args.epochs
                    schedule    = args.schedule
            elif phase == 'finetune':
                start_epoch = args.epochs - args.fepochs
                end_epoch   = args.epochs
                schedule    = args.fschedule
            for epoch in range(start_epoch, end_epoch):
                adjust_learning_rate(optimizer, args.lrd, epoch-start_epoch, schedule)
                if phase == 'init':
                    task_info = (tasks[t], class_maps[t])

                    res = train0(train_loader, model, criterion['cls'], dw, optimizer, task_info, t, epoch)
                    if result_to_tb and (t == 0):
                        write_to_tb(writer, res, phase='train', epoch=t*args.epochs+epoch)

                    res = test0(test_loader, model, criterion['cls'], task_info, t, epoch)
                    if result_to_tb and (t == 0):
                        write_to_tb(writer, res, phase='test', epoch=t*args.epochs+epoch)

                else:
                    task_info = (tasks, class_maps, prev, prev_map, seen, seen_map)
                    res = train(train_loader, model_dict, criterion, dw, optimizer, p_knows_t, loss_for_t, task_info, t, epoch)
                    if result_to_tb:
                        write_to_tb(writer, res, phase='train', epoch=t*args.epochs+epoch)

                    res = test(test_loader, model, criterion['cls'], task_info, t, epoch)
                    if result_to_tb:
                        write_to_tb(writer, res, phase='test', epoch=t*args.epochs+epoch)

        # update coreset
        if args.coreset > 0:
            train_dataset.update_coreset(args.coreset, seen)

        print('time for task {:d}: {:.3f}'.format(t, time.time() - start_time))

        # save
        model = model_dict['m'] if t > 0 else model_dict['c']
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)
        accs.append(writer.scalar_dict[writer_path + '/test/acc/mean'][-1][2])
        acc_cls = [writer.scalar_dict[writer_path + '/test/acc/{}'.format(s)][-1][2] for s in range(t+1)]
        acc_cls.extend([0.]*(num_tasks-t-1))
        accs_cls.append(acc_cls)
        with open('{}/p_{}.txt'.format(save_dir, save_str), 'a') as f:
            f.write('{}\t'.format(accs[-1]))
        if t > 0:
            with open('{}/q_{}.txt'.format(save_dir, save_str), 'a') as f:
                f.write('\t'.join('{}'.format(acc) for acc in accs_cls[-1]))
                f.write('\n')
        if writer is not None:
            writer.export_scalars_to_json(json_path)
        if t == 0:
            print('training init model complete')
            break
    if writer is not None:
        writer.close()
