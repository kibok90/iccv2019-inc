import os
import argparse
import time

import numpy as np
import h5py
import PIL.ImageFile
from PIL import Image

# some files have > 1MB metadata, e.g., n03452741/n03452741_4785.png
# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html?highlight=decompression#png
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Commands')

parser.add_argument('-n', '--num-samples', type=int, default=1000000, metavar='N',
                    help='number of samples in the external dataset')
parser.add_argument('-s', '--seeds', type=int, nargs='+', default=list(range(20)), metavar='N+',
                    help='random seeds to select the data to be in the external dataset')

args = parser.parse_args()
print(args)

num_samples = args.num_samples
seeds = args.seeds

img_root   = 'imagenet/fall11_whole_resized/'
train_root = 'imagenet/ilsvrc2012_resized/'
data_root  = 'data/imagenet/'
aux_root   = 'imagenet/'
if not os.path.isdir(data_root):
    os.makedirs(data_root)
if not os.path.isdir(aux_root):
    os.makedirs(aux_root)

# num_images = 14197122 # 21841 classes # essentially 14197060

num_train = 500
num_test = 100
num_in = num_train + num_test
num_out = num_samples
num_classes = 1000


##################
# ImageNet wnids #
##################

wnids_in_path = aux_root + 'wnids_in.npy'
if os.path.isfile(wnids_in_path):
    print('load ' + wnids_in_path)
    wnids_in = np.load(wnids_in_path).tolist()
else:
    wnids_in = sorted(os.listdir(train_root))
    np.save(wnids_in_path, wnids_in)

wnids_out_path = aux_root + 'wnids_out.npy'
if os.path.isfile(wnids_out_path):
    print('load ' + wnids_out_path)
    wnids_out = np.load(wnids_out_path).tolist()
else:
    wnids_out = sorted(os.listdir(img_root))
    np.save(wnids_out_path, wnids_out)

# [0,1000): in, [1000,21842): out\in ; note that "teddy bear" in "in" is not in "out"
wnids = wnids_in + sorted(set(wnids_out)-set(wnids_in))

filenames_in_path = aux_root + 'filenames_in.txt'
labels_in_path = aux_root + 'labels_in.npy'
if os.path.isfile(filenames_in_path):
    filenames_in = open(filenames_in_path, 'r').read().strip().splitlines()
    labels_in = np.load(labels_in_path)
else:
    filenames_in = []
    labels_in = []
    start_time = time.time()
    for k, wnid in enumerate(wnids_in):
        if k % 100 == 0:
            print('wnids in  {:5d}/{:5d} {:10.3f}'.format(k, len(wnids_in), time.time() - start_time))
        wnid_path = train_root + wnid
        if os.path.isdir(wnid_path):
            img_list = ['{}/{}'.format(wnid, filename) for filename in sorted(os.listdir(wnid_path))]
            filenames_in.extend(img_list)
            labels_in.extend([wnids.index(wnid)]*len(img_list))
    with open(filenames_in_path, 'w') as f:
        f.write('\n'.join(filenames_in))
    np.save(labels_in_path, labels_in)
    labels_in = np.array(labels_in)
filenames_in = np.array(filenames_in)

filenames_out_path = aux_root + 'filenames_out.txt'
labels_out_path = aux_root + 'labels_out.npy'
if os.path.isfile(filenames_out_path):
    filenames_out = open(filenames_out_path, 'r').read().strip().splitlines()
    labels_out = np.load(labels_out_path)
else:
    filenames_out = []
    labels_out = []
    start_time = time.time()
    for k, wnid in enumerate(wnids_out):
        if k % 100 == 0:
            print('wnids out {:5d}/{:5d} {:10.3f}'.format(k, len(wnids_out), time.time() - start_time))
        wnid_path = img_root + wnid
        if os.path.isdir(wnid_path):
            img_list = ['{}/{}'.format(wnid, filename) for filename in sorted(os.listdir(wnid_path))]
            filenames_out.extend(img_list)
            labels_out.extend([wnids.index(wnid)]*len(img_list))
    with open(filenames_out_path, 'w') as f:
        f.write('\n'.join(filenames_out))
    np.save(labels_out_path, labels_out)
    labels_out = np.array(labels_out)
filenames_out = np.array(filenames_out)

num_images = len(labels_out)
print(num_images)


####################
# Sampling indexes #
####################

locs_train_path = aux_root + 'imagenet_locs_train.npy'
locs_test_path  = aux_root + 'imagenet_locs_test.npy'

if os.path.isfile(locs_train_path):
    print('load ' + locs_train_path)
    print('load ' + locs_test_path)
    locs_train = np.load(locs_train_path)
    locs_test = np.load(locs_test_path)
else:
    locs_train = []
    locs_test = []
    start_time = time.time()
    np.random.seed(0)
    for k, wnid in enumerate(wnids_in):
        if k % 100 == 0:
            print('trte {:5d}/{:5d} {:10.3f}'.format(k, len(wnids_in), time.time() - start_time))
        locs_in_k = np.isin(labels_in, k).nonzero()[0]
        locs_trte_k = locs_in_k[np.random.choice(len(locs_in_k), num_in, replace=False)]
        locs_train.extend(locs_trte_k[:num_train])
        locs_test.extend(locs_trte_k[num_train:])
    
    np.save(locs_train_path, locs_train)
    np.save(locs_test_path, locs_test)

locs_out = (labels_out >= len(wnids_in)).nonzero()[0]
print(len(locs_out))


#####################
# Sampling ImageNet #
#####################

start_time = time.time()
for seed in seeds:
    locs_out_seed_path = aux_root + 'imagenet_locs_out_{}_{}.npy'.format(num_out, seed)
    
    if os.path.isfile(locs_out_seed_path):
        print('load ' + locs_out_seed_path)
        locs_out_chosen = np.load(locs_out_seed_path)
    else:
        np.random.seed(seed)
        locs_out_chosen = locs_out[np.random.choice(len(locs_out), num_out, replace=False)]
        np.save(locs_out_seed_path, locs_out_chosen)
    
    print('seed {:1d}: {:10.3f}'.format(seed, time.time() - start_time))
    
    for h, locs in enumerate([locs_train, locs_test, locs_out_chosen]):
        if (seed != 0) and (h != 2): continue
        if   h == 0: save_path = data_root + 'imagenet_train_{}.h5'.format(num_train)
        elif h == 1: save_path = data_root + 'imagenet_test_{}.h5'.format(num_test)
        elif h == 2: save_path = data_root + 'imagenet_out_{}_{}.h5'.format(num_out, seed)
        print(save_path)
        print(len(locs))
        
        if os.path.isfile(save_path):
            continue
        
        if h in [0,1]:
            root = train_root
            filenames = filenames_in
            labels_in_out = labels_in
            num_max = num_train if h == 0 else num_test
            num_max *= num_classes
        else:
            root = img_root
            filenames = filenames_out
            labels_in_out = labels_out
            num_max = num_out
        
        data   = []
        labels = []
        for i, ind in enumerate(locs):
            if i % 10000 == 0:
                print('seed {:1d} {:7d}/{:7d}: {:10.3f}'.format(seed, i, num_max, time.time() - start_time))
            
            filename = filenames[ind]
            label    = labels_in_out[ind]
            try:
                im = Image.open(root + filename)
                data.append(np.array(im))
                labels.append(label)
            except OSError as err:
                print("Couldn't load: %s" % (root + filename))
                with open("log.txt", "a") as f:
                    f.write("Couldn't load: %s" % (root + filename))
        
        data = np.stack(data, axis=0)
        labels = np.array(labels)
        
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('data',    data=data  )
            f.create_dataset('labels',  data=labels)
