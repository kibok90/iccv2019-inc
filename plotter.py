import os
import argparse
import numpy as np

dataset_cands = ['cifar100', 'imagenet']
ex_dataset_cands = ['none', 'tiny', 'imagenet']
exp_cands = ['custom', 'bar', 'time', 't1', 't2', 't3', 't4', 't5']

parser = argparse.ArgumentParser(description='Commands')

parser.add_argument('-s', '--save-dir', type=str, default='res', metavar='DIR',
                    help='model directory')
parser.add_argument('-f', '--fig-dir', type=str, default='fig', metavar='DIR',
                    help='figure directory')
parser.add_argument('-d', '--dataset', type=str, default='cifar100', metavar='DSET',
                    choices=dataset_cands,
                    help='dataset: ' + ' | '.join(dataset_cands) + ' (default: cifar100)')
parser.add_argument('-e', '--ex-dataset', type=str, default='tiny', metavar='DSET',
                    choices=ex_dataset_cands,
                    help='external dataset: ' + ' | '.join(ex_dataset_cands) + ' (default: tiny)')
parser.add_argument('-t', '--task-size', type=int, nargs='+', default=[10, 10], metavar='N+',
                    help='number of initial classes and incremental classes (default: 10 10)')
parser.add_argument('--exp', type=str, default='custom', metavar='EXP',
                    choices=exp_cands,
                    help='experiment for evaluation: ' + ' | '.join(exp_cands) + ' (default: custom)')
parser.add_argument('-v', '--val', action='store_true', default=False,
                    help='validation with imagenet seed=0 (default: False)')
parser.add_argument('--seeds', type=int, nargs='+', default=[], metavar='N+',
                    help='manual random seed (default: all)')

args = parser.parse_args()
print(args)

save_dir = args.save_dir
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
fig_dir = args.fig_dir
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

# figure properties
if args.exp in ['bar', 'time']:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    xname_time = 'Number of classes'
    xname_bar = 'Task size $\\times$ Number of tasks'
    yname_acc = 'Accuracy (%)'
    yname_fgt = 'Forgetting (%)'

    figsize = (9.6,5.4)
    bbox_inches = matplotlib.transforms.Bbox([(0.,0.), figsize])
    bbox_inches_extended = matplotlib.transforms.Bbox([(0.,0.), (figsize[0]+3.2,figsize[1])])
    fig_margins = {'top': 0.97, 'bottom': 0.15, 'left': 0.12, 'right': 0.97}
    sns.set(font_scale=2., rc={'figure.figsize': figsize}, style='whitegrid')

    # 0 blue, 1 orange, 2 green, 3 red, 4 purple, 5 brown, 6 pink, 7 grey, 8 yellow, 9 sky
    color_order = [0, 1, 2, 4, 5, 3]
    color_palette = sns.color_palette('deep') # default
    color_palette_pastel = sns.color_palette('pastel')
    color_palette_dark = sns.color_palette('dark')
    color_palette = [color_palette[i] for i in color_order]
    color_palette_pastel = [color_palette_pastel[i] for i in color_order]
    color_palette_dark = [color_palette_dark[i] for i in color_order]

    markers = ['^', 'v', 's', 'p', 'o', 'D']
    dashes = [(1, 1), (5, 1, 1, 1), (3, 1, 1.5, 1), (4, 1.5), '', '']
    linewidth = 3.
    markersize = 9.

ee = 1e-8
num_classes = 100

# seeds
if len(args.seeds) > 0:
    seeds = args.seeds
elif args.val:
    seeds = [0]
elif args.dataset == 'imagenet':
    seeds = range(1,10)
else:
    seeds = range(10)

# model names
if args.exp == 'bar':
    model_names = [
        '{dataset}_{seed}_t={base}_{inc}_cs=2000_ref=L_kdr=1.0_T=2.0_bd=none',
        '{dataset}_{seed}_co=1.0_t={base}_{inc}_cs=2000_ref=LC_kdr=1.0_T=2.0_bd=none',
        '{dataset}_{seed}_t={base}_{inc}_cs=2000_ref=L_kdr=1.0_T=2.0_bd=dset_ft=all',
        '{dataset}_{seed}_co=1.0_t={base}_{inc}_cs=2000_ref=PC_kdr=1.0_T=2.0_bd=dw_ft=cls',
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=L_kdr=1.0_T=2.0_bd=none_exr=1.0_ood=0.7',
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=LC_kdr=1.0_T=2.0_bd=none_exr=1.0_ood=0.7',
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=L_kdr=1.0_T=2.0_bd=dset_ft=all_exr=1.0_ood=0.7',
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=PCQ_kdr=1.0_T=2.0_qT=1.0_bd=dw_ft=cls_exr=1.0_ood=0.7',
    ]
elif args.exp == 'time':
    model_names = [
        '{dataset}_{seed}_t={base}_{inc}_cs=2000_ref=L_kdr=1.0_T=2.0_bd=none',
        '{dataset}_{seed}_co=1.0_t={base}_{inc}_cs=2000_ref=LC_kdr=1.0_T=2.0_bd=none',
        '{dataset}_{seed}_t={base}_{inc}_cs=2000_ref=L_kdr=1.0_T=2.0_bd=dset_ft=all',
        '{dataset}_{seed}_co=1.0_t={base}_{inc}_cs=2000_ref=PC_kdr=1.0_T=2.0_bd=dw_ft=cls',
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=PCQ_kdr=1.0_T=2.0_qT=1.0_bd=dw_ft=cls_exr=1.0_ood=0.7',
    ]
elif args.exp == 't1':
    model_names = [
        '{dataset}_{seed}_t={base}_{inc}_cs=2000_bd=none',
        '{dataset}_{seed}_t={base}_{inc}_cs=2000_ref=L_kdr=1.0_T=2.0_bd=none',
        '{dataset}_{seed}_co=1.0_t={base}_{inc}_cs=2000_ref=LC_kdr=1.0_T=2.0_bd=none',
        '{dataset}_{seed}_t={base}_{inc}_cs=2000_ref=L_kdr=1.0_T=2.0_bd=dset_ft=all',
        '{dataset}_{seed}_co=1.0_t={base}_{inc}_cs=2000_ref=PC_kdr=1.0_T=2.0_bd=dw_ft=cls',
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=PCQ_kdr=1.0_T=2.0_qT=1.0_bd=dw_ft=cls_exr=1.0_ood=0.7',
    ]
elif args.exp == 't2':
    model_names = [
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=P_kdr=1.0_T=2.0_bd=dw_ft=cls_exr=1.0_ood=0.7',
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=PC_kdr=1.0_T=2.0_bd=dw_ft=cls_exr=1.0_ood=0.7',
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=Q_kdr=1.0_qT=1.0_bd=dw_ft=cls_exr=1.0_ood=0.7',
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=PCQ_kdr=1.0_T=2.0_qT=1.0_bd=dw_ft=cls_exr=1.0_ood=0.7',
    ]
elif args.exp == 't3':
    model_names = [
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=P_kdr=1.0_T=2.0_bd=dw_ft=cls_exr=1.0_ood=0.7',
        '{dataset}_{seed}_e={ex_dataset}_co=0.0_t={base}_{inc}_cs=2000_ref=PO_kdr=1.0_T=2.0_bd=dw_ft=cls_exr=1.0_ood=0.7',
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=PO_kdr=1.0_T=2.0_bd=dw_ft=cls_exr=1.0_ood=0.7',
        '{dataset}_{seed}_e={ex_dataset}_co=0.0_t={base}_{inc}_cs=2000_ref=PC_kdr=1.0_T=2.0_bd=dw_ft=cls_exr=1.0_ood=0.7',
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=PC_kdr=1.0_T=2.0_bd=dw_ft=cls_exr=1.0_ood=0.7',
    ]
elif args.exp == 't4':
    model_names = [
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=PCQ_kdr=1.0_T=2.0_qT=1.0_bd=none_exr=1.0_ood=0.7',
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=PCQ_kdr=1.0_T=2.0_qT=1.0_bd=dw_exr=1.0_ood=0.7',
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=PCQ_kdr=1.0_T=2.0_qT=1.0_bd=dset_ft=all_exr=1.0_ood=0.7',
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=PCQ_kdr=1.0_T=2.0_qT=1.0_bd=dw_ft=cls_exr=1.0_ood=0.7',
    ]
elif args.exp == 't5':
    model_names = [
        '{dataset}_{seed}_co=1.0_t={base}_{inc}_cs=2000_ref=PC_kdr=1.0_T=2.0_bd=dw_ft=cls',
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=PCQ_kdr=1.0_T=2.0_qT=1.0_bd=dw_ft=cls_exr=1.0_ood=1.0',
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=PCQ_kdr=1.0_T=2.0_qT=1.0_bd=dw_ft=cls_exr=1.0_ood=0.0',
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=PCQ_kdr=1.0_T=2.0_qT=1.0_bd=dw_ft=cls_exr=1.0_oodp=0.7',
        '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}_{inc}_cs=2000_ref=PCQ_kdr=1.0_T=2.0_qT=1.0_bd=dw_ft=cls_exr=1.0_ood=0.7',
    ]
else:
    raise NotImplementedError('put your custom models in the code')

# model names in figure
if args.exp == 'bar':
    model_names_in_fig = ['LwF', 'DR', 'E2E', 'GD (Ours)'] * 2
    num_models = sum([1 for model_name in model_names if '_e=' not in model_name])
elif args.exp == 'time':
    model_names_in_fig = ['LwF', 'DR', 'E2E', 'GD (Ours)', 'GD+ext (Ours)']
    num_models = len(model_names)
else:
    model_names_in_fig = model_names

if args.exp == 'bar':
    task_sizes = [[5, 5], [10, 10], [20, 20]]
    accs_bar_pd    = pd.DataFrame(columns=['Model', xname_bar, yname_acc, 'Seed'])
    fgts_bar_pd    = pd.DataFrame(columns=['Model', xname_bar, yname_fgt, 'Seed'])
    accs_ex_bar_pd = pd.DataFrame(columns=['Model', xname_bar, yname_acc, 'Seed'])
    fgts_ex_bar_pd = pd.DataFrame(columns=['Model', xname_bar, yname_fgt, 'Seed'])
else:
    task_sizes = [args.task_size]

for task_size in task_sizes:
    if args.exp in ['bar', 'time']:
        accs_pd = pd.DataFrame(columns=['Model', xname_time, yname_acc, 'Seed'])
        fgts_pd = pd.DataFrame(columns=['Model', xname_time, yname_fgt, 'Seed'])

    num_tasks_per = []
    p = 0
    while p < num_classes:
        inc = task_size[1] if p > 0 else task_size[0]
        num_tasks_per.append(inc)
        p += inc
    task_weight = np.array(num_tasks_per) / num_classes
    num_tasks = len(num_tasks_per)

    accs_history = np.zeros([len(model_names), len(seeds)], dtype=float)
    base_acc_history = np.zeros([len(model_names), len(seeds)], dtype=float)
    for m, model_name in enumerate(model_names):
        for d, seed in enumerate(seeds):
            succeed = False

            pos = model_name.find('co=')
            if pos < 0:
                co = 0.
            else:
                pot = model_name.find('_', pos+3)
                co = float(model_name[pos+3:pot])
            if (co > 0.) and ('e=' in model_name):
                base_name = '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}'
            else:
                base_name = '{dataset}_{seed}_t={base}'

            # base acc
            base_name_formatted = base_name.format(dataset=args.dataset, seed=seed, ex_dataset=args.ex_dataset, base=task_size[0])
            base_path = os.path.join(save_dir, 'p_{}.txt'.format(base_name_formatted))
            if os.path.isfile(base_path):
                base_acc = np.loadtxt(base_path, dtype='float')
                base_acc_history[m,d] = base_acc

            # inc acc
            model_name_formatted = model_name.format(dataset=args.dataset, seed=seed, ex_dataset=args.ex_dataset, base=task_size[0], inc=task_size[1])
            path = os.path.join(save_dir, 'p_{}.txt'.format(model_name_formatted))
            if os.path.isfile(path):
                accs = np.loadtxt(path, dtype='float')
                accs_history[m,d] = accs.mean()
                accs = np.concatenate([base_acc[None], accs])
                print('load {}: {}'.format(m, accs[1:].mean()))
                succeed = True

            if succeed:
                if args.exp == 'bar':
                    this_pd = pd.DataFrame(data={xname_bar : '{} $\\times$ {}'.format(task_size[1], num_classes // task_size[1]),
                                                 yname_acc : accs_history[m,d:d+1],
                                                 'Model'   : '{}'.format(model_names_in_fig[m]),
                                                 'Seed'    : seed,
                                                })
                    if '_e=' in model_name:
                        accs_ex_bar_pd = accs_ex_bar_pd.append(this_pd, sort=False)
                    else:
                        accs_bar_pd = accs_bar_pd.append(this_pd, sort=False)
                elif args.exp == 'time':
                    this_pd = pd.DataFrame(data={xname_time: task_size[0]+task_size[1]*np.arange(0,len(accs)),
                                                 yname_acc : accs,
                                                 'Model'   : '{}'.format(model_names_in_fig[m]),
                                                 'Seed'    : seed,
                                                })
                    accs_pd = accs_pd.append(this_pd, sort=False)
            else:
                print('fail {}: {}'.format(m, -1))

    print('acc mean:')
    for val in accs_history.mean(axis=1):
        print(val)
    if len(seeds) > 1:
        print('acc std:')
        for val in accs_history.std(axis=1, ddof=1):
            print(val)

    # class-wise accuracy [model, seed, stage, task]
    accs_cls_history = np.zeros([len(model_names), len(seeds), num_tasks, num_tasks], dtype=float)
    for m, model_name in enumerate(model_names):
        for d, seed in enumerate(seeds):
            succeed = False

            pos = model_name.find('co=')
            if pos < 0:
                co = 0.
            else:
                pot = model_name.find('_', pos+3)
                co = float(model_name[pos+3:pot])
            if (co > 0.) and ('e=' in model_name):
                base_name = '{dataset}_{seed}_e={ex_dataset}_co=1.0_t={base}'
            else:
                base_name = '{dataset}_{seed}_t={base}'

            # base acc
            base_name_formatted = base_name.format(dataset=args.dataset, seed=seed, ex_dataset=args.ex_dataset, base=task_size[0])
            base_path = os.path.join(save_dir, 'p_{}.txt'.format(base_name_formatted))
            if os.path.isfile(base_path):
                base_acc = np.loadtxt(base_path, dtype='float')
                accs_cls_history[m,d,0,0] = base_acc

            # inc acc from q
            model_name_formatted = model_name.format(dataset=args.dataset, seed=seed, ex_dataset=args.ex_dataset, base=task_size[0], inc=task_size[1])
            path = os.path.join(save_dir, 'q_{}.txt'.format(model_name_formatted))
            if os.path.isfile(path):
                inc_acc = np.loadtxt(path, dtype='float')
                accs_cls_history[m,d,1:] = inc_acc
                succeed = True

    # compute forgetting
    fgts_st_history = np.zeros_like(accs_cls_history)
    fgts_s_history  = np.zeros(fgts_st_history.shape[:-1], dtype=float)
    for t in range(num_tasks):
        fgts_st_history[:,:,t:,t] = accs_cls_history[:,:,t,t][:,:,None] - accs_cls_history[:,:,t:,t]
    for t in range(num_tasks):
        fgts_s_history[:,:,t] = (fgts_st_history[:,:,t,:t] * task_weight[:t]).sum(axis=2)
    fgts_history = fgts_s_history.mean(axis=2)


    if args.exp == 'bar':
        for m, model_name in enumerate(model_names):
            for d, seed in enumerate(seeds):
                this_pd = pd.DataFrame(data={xname_bar : '{} $\\times$ {}'.format(task_size[1], num_classes // task_size[1]),
                                             yname_fgt : fgts_history[m,d:d+1],
                                             'Model'   : '{}'.format(model_names_in_fig[m]),
                                             'Seed'    : seed
                                            })
                if '_e=' in model_name:
                    fgts_ex_bar_pd = fgts_ex_bar_pd.append(this_pd, sort=False)
                else:
                    fgts_bar_pd = fgts_bar_pd.append(this_pd, sort=False)
    elif args.exp == 'time':
        for m, model_name in enumerate(model_names):
            for d, seed in enumerate(seeds):
                this_pd = pd.DataFrame(data={xname_time: task_size[0]+task_size[1]*np.arange(0,num_tasks),
                                             yname_fgt : fgts_s_history[m,d],
                                             'Model'   : '{}'.format(model_names_in_fig[m]),
                                             'Seed'    : seed,
                                            })
                fgts_pd = fgts_pd.append(this_pd, sort=False)

    print('fgt mean:')
    for val in fgts_history.mean(axis=1):
        print(val)
    if len(seeds) > 1:
        print('fgt std:')
        for val in fgts_history.std(axis=1, ddof=1):
            print(val)

    if args.exp == 'time':
        if 'acc_plot' in dir():
            acc_plot.clear()
        sns.despine(top=True,right=True)
        acc_plot = sns.lineplot(data=accs_pd, x=xname_time, y=yname_acc, hue='Model', style='Model', ci='sd',
                                palette=color_palette[:num_models], markers=markers[:num_models], dashes=dashes[:num_models],
                                linewidth=linewidth, markersize=markersize)
        if args.dataset == 'cifar100':
            if   task_size[0] == 5:
                ymin, ymax = 35, 100
            elif task_size[0] == 10:
                ymin, ymax = 40, 95
            elif task_size[0] == 20:
                ymin, ymax = 45, 90
            else:
                ymin, ymax = 0, 100
        else:
            if   task_size[0] == 5:
                ymin, ymax = 25, 90
            elif task_size[0] == 10:
                ymin, ymax = 30, 85
            elif task_size[0] == 20:
                ymin, ymax = 35, 80
            else:
                ymin, ymax = 0, 100
        acc_plot.set(xlim=(task_size[0],100.1), xticks=np.arange(0,101,task_size[1]), ylim=(ymin, ymax), yticks=np.arange(ymin,ymax+1,5))

        acc_plot.legend_.remove()
        # h, l = acc_plot.get_legend_handles_labels()
        # for i, line in enumerate(h):
            # line.set_linewidth(linewidth)
            # line.set_markersize(markersize)
        # acc_plot.legend(handles=h[1:], labels=l[1:])

        g = acc_plot.get_figure()
        g.subplots_adjust(**fig_margins)
        g.savefig('{}/acc_time_{}_{}_{}.png'.format(fig_dir, args.dataset, task_size[0], task_size[1]), bbox_inches=bbox_inches)
        g.clf()

        if 'fgt_plot' in dir():
            fgt_plot.clear()
        sns.despine(top=True,right=True)
        fgt_plot = sns.lineplot(data=fgts_pd, x=xname_time, y=yname_fgt, hue='Model', style='Model', ci='sd',
                                palette=color_palette[:num_models], markers=markers[:num_models], dashes=dashes[:num_models],
                                linewidth=linewidth, markersize=markersize)
        if args.dataset == 'cifar100':
            if   task_size[0] == 5:
                ymin, ymax = 0, 55
            elif task_size[0] == 10:
                ymin, ymax = 0, 45
            elif task_size[0] == 20:
                ymin, ymax = 0, 40
            else:
                ymin, ymax = 0, 100
        else:
            if   task_size[0] == 5:
                ymin, ymax = 0, 60
            elif task_size[0] == 10:
                ymin, ymax = 0, 50
            elif task_size[0] == 20:
                ymin, ymax = 0, 40
            else:
                ymin, ymax = 0, 100
        fgt_plot.set(xlim=(task_size[0],100.1), xticks=np.arange(0,101,task_size[1]), ylim=(ymin, ymax), yticks=np.arange(ymin,ymax+1,5))
        h, l = fgt_plot.get_legend_handles_labels()
        for i, line in enumerate(h):
            line.set_linewidth(linewidth)
            line.set_markersize(markersize)
        fgt_plot.legend(handles=h[-num_models:], labels=l[-num_models:], loc='center right', bbox_to_anchor=(1.44, 0.5), ncol=1)

        g = fgt_plot.get_figure()
        g.subplots_adjust(**fig_margins)
        g.savefig('{}/fgt_time_{}_{}_{}.png'.format(fig_dir, args.dataset, task_size[0], task_size[1]), bbox_inches=bbox_inches_extended)
        g.clf()

bar_error_bar = False
if args.exp == 'bar':

    if 'acc_bar_plot_pastel' in dir():
        acc_bar_plot_pastel.clear()
    if 'acc_bar_plot' in dir():
        acc_bar_plot.clear()
    sns.despine(top=True,right=True)
    if bar_error_bar:
        acc_bar_plot_pastel = sns.barplot(data=accs_ex_bar_pd, x=xname_bar, y=yname_acc, hue='Model', ci='sd',
                                          palette=color_palette_pastel[:num_models], capsize=.05, errcolor='.63')
        acc_bar_plot = sns.barplot(data=accs_bar_pd, x=xname_bar, y=yname_acc, hue='Model', ci='sd',
                                   palette=color_palette[:num_models], capsize=.05)
    else:
        acc_bar_plot_pastel = sns.barplot(data=accs_ex_bar_pd, x=xname_bar, y=yname_acc, hue='Model', ci=None,
                                          palette=color_palette_pastel[:num_models])
        acc_bar_plot = sns.barplot(data=accs_bar_pd, x=xname_bar, y=yname_acc, hue='Model', ci=None,
                                   palette=color_palette[:num_models])
    if args.dataset == 'cifar100':
        ymin, ymax = 50, 70
    else:
        ymin, ymax = 40, 60
    acc_bar_plot.set(ylim=(ymin, ymax), yticks=np.arange(ymin,ymax+1,5))
    h, l = acc_bar_plot.get_legend_handles_labels()
    for i in range(num_models):
        for bar, bar_pastel in zip(h[i], h[i+num_models]):
            xy = (bar.get_x() + bar.get_width() / 2., bar.get_y() + bar.get_height())
            xytext = (bar_pastel.get_x() + bar_pastel.get_width() / 2., bar_pastel.get_y() + bar_pastel.get_height())
            headlength = np.minimum(xy[1] - xytext[1], 2.) * 10.
            acc_bar_plot.annotate('', xy, xytext, arrowprops=dict(facecolor=color_palette_dark[i], width=10., headwidth=20., headlength=headlength))

    acc_bar_plot.legend_.remove()
    # acc_bar_plot.legend(handles=h[-num_models:], labels=l[-num_models:])

    g = acc_bar_plot.get_figure()
    g.subplots_adjust(**fig_margins)
    g.savefig('{}/acc_bar_{}.png'.format(fig_dir, args.dataset), bbox_inches=bbox_inches)
    g.clf()

    if 'fgt_bar_plot_pastel' in dir():
        fgt_bar_plot_pastel.clear()
    if 'fgt_bar_plot' in dir():
        fgt_bar_plot.clear()
    sns.despine(top=True,right=True)
    if bar_error_bar:
        fgt_bar_plot_pastel = sns.barplot(data=fgts_bar_pd, x=xname_bar, y=yname_fgt, hue='Model', ci='sd',
                                          palette=color_palette_pastel, capsize=.05, errcolor='.63')
        fgt_bar_plot = sns.barplot(data=fgts_ex_bar_pd, x=xname_bar, y=yname_fgt, hue='Model', ci='sd',
                                   palette=color_palette, capsize=.05)
    else:
        fgt_bar_plot_pastel = sns.barplot(data=fgts_bar_pd, x=xname_bar, y=yname_fgt, hue='Model', ci=None,
                                          palette=color_palette_pastel)
        fgt_bar_plot = sns.barplot(data=fgts_ex_bar_pd, x=xname_bar, y=yname_fgt, hue='Model', ci=None,
                                   palette=color_palette)
    if args.dataset == 'cifar100':
        ymin, ymax = 0, 25
    else:
        ymin, ymax = 0, 25
    fgt_bar_plot.set(ylim=(ymin, ymax), yticks=np.arange(ymin,ymax+1,5))
    h, l = fgt_bar_plot.get_legend_handles_labels()
    for i in range(num_models):
        for bar_pastel, bar in zip(h[i], h[i+num_models]):
            xy = (bar.get_x() + bar.get_width() / 2., bar.get_y() + bar.get_height())
            xytext = (bar_pastel.get_x() + bar_pastel.get_width() / 2., bar_pastel.get_y() + bar_pastel.get_height())
            headlength = np.minimum(xytext[1] - xy[1], 2.) * 10.
            fgt_bar_plot.annotate('', xy, xytext, arrowprops=dict(facecolor=color_palette_dark[i], width=10., headwidth=20., headlength=headlength))
    fgt_bar_plot.legend(handles=h[-num_models:], labels=l[-num_models:], loc='center right', bbox_to_anchor=(1.345, 0.5), ncol=1)

    g = fgt_bar_plot.get_figure()
    g.subplots_adjust(**fig_margins)
    g.savefig('{}/fgt_bar_{}.png'.format(fig_dir, args.dataset), bbox_inches=bbox_inches_extended)
    g.clf()
