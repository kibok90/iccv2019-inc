import os
import argparse
import time

import numpy as np
import scipy.io
import h5py

import datasets

parser = argparse.ArgumentParser(description='Commands')

parser.add_argument('-n', '--num-samples', type=int, default=1000000, metavar='N',
                    help='number of samples in the external dataset')
parser.add_argument('-s', '--seeds', type=int, nargs='+', default=list(range(20)), metavar='N+',
                    help='random seeds to select the data to be in the external dataset')

args = parser.parse_args()
print(args)

num_samples = args.num_samples
seeds = args.seeds

data_root = 'data/tiny/'
aux_root = 'tiny/'
if not os.path.isdir(data_root):
    os.makedirs(data_root)
if not os.path.isdir(aux_root):
    os.makedirs(aux_root)

cifar_file_path = data_root + 'cifar_indexes'
cifar_indexes = open(cifar_file_path, 'r').read().strip().splitlines()
cifar_indexes = list(map(lambda x:x-1, map(int, cifar_indexes)))
locs_cifar = np.array(cifar_indexes)
locs_cifar = locs_cifar[locs_cifar >= 0]

num_tinyimages = 79302017
num_train_data = 50000
num_in = 500
num_out = num_samples
num_classes = 100
data_unit = 3072
meta_unit = 768
data_file_path = data_root + 'tiny_images.bin'
meta_file_path = data_root + 'tiny_metadata.bin'


################################################
# metadata of CIFAR-100 images from TinyImages #
################################################

cifar100_from_tinyimages_path = aux_root + 'cifar100_from_tinyimages.npy'
if os.path.isfile(cifar100_from_tinyimages_path):
    print('load ' + cifar100_from_tinyimages_path)
    meta = np.load(cifar100_from_tinyimages_path).item()['meta']
else:
    data_file = open(data_file_path, 'rb')
    meta_file = open(meta_file_path, 'rb')
    start_time = time.time()
    data  = []
    meta  = []
    for i, ind in enumerate(cifar_indexes):
        if i % 100 == 0:
            print('metadata {:5d}/{:5d}: {:10.3f}'.format(i, len(cifar_indexes), time.time() - start_time))
        if ind < 0:
            data.append(np.zeros([32,32,3], dtype=np.uint8))
            meta.append(b' ' * meta_unit)
        else:
            data_file.seek(ind*data_unit)
            buffer = data_file.read(data_unit)
            data.append(np.frombuffer(buffer, dtype=np.uint8).copy().reshape(32,32,3, order='F'))
            
            meta_file.seek(ind*meta_unit)
            buffer = meta_file.read(meta_unit)
            meta.append(buffer)
    data_file.close()
    meta_file.close()
    data = np.stack(data, axis=0)
    np.save(cifar100_from_tinyimages_path, {'data': data, 'meta': meta})


#########################################################
# TinyImages class to the number of images in CIFAR-100 #
#########################################################

words_path = aux_root + 'words_cifar100.npy'
if os.path.isfile(words_path):
    print('load ' + words_path)
    tmp = np.load(words_path).item()
    train_words = tmp['train_words']
    test_words  = tmp['test_words']
    del tmp
else:
    train_dataset = datasets.iCIFAR100('data/cifar100', 'cifar100', train=True)
    test_dataset  = datasets.iCIFAR100('data/cifar100', 'cifar100', train=False)
    train_words = []
    for k in range(num_classes):
        train_words.append({})
        inds = (train_dataset.targets == k).nonzero()[0]
        for i in inds:
            word = meta[i][:80].strip()
            if word in train_words[k]:
                train_words[k][word] += 1
            else:
                train_words[k].update({word: 1})
    test_words = []
    for k in range(num_classes):
        test_words.append({})
        inds = (test_dataset.targets == k).nonzero()[0] + num_train_data
        for i in inds:
            word = meta[i][:80].strip()
            if word in test_words[k]:
                test_words[k][word] += 1
            else:
                test_words[k].update({word: 1})
    np.save(words_path, {'train_words': train_words, 'test_words': test_words})


#####################################
# TinyImages words and class labels #
#####################################

tiny_words_path = aux_root + 'tiny_words.npy'
if os.path.isfile(tiny_words_path):
    print('load ' + tiny_words_path)
    twords = np.load(tiny_words_path).tolist()
else:
    tiny_index = scipy.io.loadmat(data_root + 'tiny_index.mat')
    twords = [tiny_index['word'][0][i][0].replace('\x1a','') for i in range(len(tiny_index['word'][0]))]
    # valid_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_,.'
    # for i, word in enumerate(twords):
        # for c in word:
            # if c not in valid_chars:
                # print(str(i) + ': ' + word)
    np.save(tiny_words_path, twords)

from_tiny_index = True
tiny_labels_path = aux_root + 'tiny_labels.npy'
tiny_words_exhaust_path = aux_root + 'tiny_words_exhaust.npy'
if os.path.isfile(tiny_labels_path):
    print('load ' + tiny_labels_path)
    tlabels = np.load(tiny_labels_path)
    if os.path.isfile(tiny_words_exhaust_path):
        print('load ' + tiny_words_exhaust_path)
        twords = np.load(tiny_words_exhaust_path).tolist()
else:
    tlabels = np.full(num_tinyimages, -1, dtype=int)
    if from_tiny_index: # confirmed that this works
        tiny_index = scipy.io.loadmat(data_root + 'tiny_index.mat')
        stats = tiny_index['num_imgs'][0]
        cumsum = 0
        for k in range(len(stats)):
            tlabels[cumsum:(cumsum+stats[k])] = k
            cumsum += stats[k]
        np.save(tiny_labels_path, tlabels)
    else: # this takes a long time
        meta_file = open(meta_file_path, 'rb')
        start_time = time.time()
        for i in range(num_tinyimages):
            if i % 100000 == 0:
                print('tiny words {:8d}/{:8d}: {:10.3f}'.format(i, num_tinyimages, time.time() - start_time))
            meta_file.seek(i*meta_unit)
            buffer = meta_file.read(meta_unit)
            word = buffer[:80].strip().decode().replace('\x1a','')
            try:
                tlabels[i] = twords.index(word)
            except ValueError:
                print('{:8d}/{:8d}: add {}'.format(i, num_tinyimages, word))
                twords.append(word)
                tlabels[i] = twords.index(word)
        meta_file.close()
        np.save(tiny_labels_path, tlabels)
        np.save(tiny_words_exhaust_path, twords)
        
        # sanity check: tlabels are in increasing order
        tlabels_sorted = tlabels.copy()
        tlabels_sorted.sort()
        assert (tlabels_sorted == tlabels).all(), meta_file_path + ' is not in increasing order'
        
        # sanity check: consistent statistics from tiny_index.mat
        stats = np.histogram(tlabels, bins=range(len(twords)+1))[0]
        tiny_index = scipy.io.loadmat(data_root + 'tiny_index.mat')
        assert (tiny_index['num_imgs'][0] == stats).all(), 'statistics are not consistent'


####################
# Sampling indexes #
####################

tiny_classes_in_path = aux_root + 'tiny_classes_in.npy'
tiny_locs_out_path = aux_root + 'tiny_locs_out.npy'

if os.path.isfile(tiny_classes_in_path):
    print('load ' + tiny_classes_in_path)
    print('load ' + tiny_locs_out_path)
    classes_in = np.load(tiny_classes_in_path)
    locs_out = np.load(tiny_locs_out_path)
else:
    classes_in = []
    classes_in_flat = []
    for h, train_test_words in enumerate([train_words, test_words]):
        for k, words in enumerate(train_test_words):
            if h == 0:
                classes_in.append([])
            for word in words:
                word = word.decode().replace('\x1a','')
                if word in twords:
                    l = twords.index(word)
                    if l not in classes_in_flat:
                        classes_in_flat.append(l)
                        classes_in[k].append(l)
    
    locs_out = ~np.isin(tlabels, classes_in_flat)
    np.save(tiny_classes_in_path, classes_in)
    np.save(tiny_locs_out_path, locs_out)


#######################
# Sampling TinyImages #
#######################

data_file = open(data_file_path, 'rb')
meta_file = open(meta_file_path, 'rb')
locs_out = locs_out.nonzero()[0]

start_time = time.time()
for seed in seeds:
    locs_out_seed_path = aux_root + 'tiny_locs_out_{}_{}.npy'.format(num_out, seed)
    
    if os.path.isfile(locs_out_seed_path):
        print('load ' + locs_out_seed_path)
        locs = np.load(locs_out_seed_path)
    else:
        np.random.seed(seed)
        locs = locs_out[np.random.choice(len(locs_out), num_out, replace=False)]
        np.save(locs_out_seed_path, locs)
    
    print('seed {:1d}: {:10.3f}'.format(seed, time.time() - start_time))
    
    save_path = data_root + 'tiny_out_{}_{}.h5'.format(num_out, seed)
    print(save_path)
    print(len(locs))
    
    if os.path.isfile(save_path):
        continue
    
    order   = locs.argsort()
    rorder  = order.argsort()
    data    = []
    tlabels = []
    for i, ind in enumerate(locs[order]):
        if i % 10000 == 0:
            print('seed {:1d} {:7d}/{:7d}: {:10.3f}'.format(seed, i, num_out, time.time() - start_time))
        data_file.seek(ind*data_unit)
        buffer = data_file.read(data_unit)
        data.append(np.frombuffer(buffer, dtype=np.uint8).copy().reshape(32,32,3, order='F'))
        
        meta_file.seek(ind*meta_unit)
        buffer = meta_file.read(meta_unit)
        word = buffer[:80].strip().decode().replace('\x1a','')
        if word in twords:
            tlabels.append(twords.index(word))
        else:
            print('seed {:1d} {:7d}/{:7d}: cannot find word {}'.format(seed, i, num_out, word))
            tlabels.append(-1)
    data    = np.stack(data, axis=0)[rorder]
    tlabels = np.array(tlabels, dtype=int)[rorder]
    labels  = np.full(len(tlabels), -1, dtype=int)
    for k, class_in in enumerate(classes_in):
        labels[np.isin(tlabels, class_in)] = k
    
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('data',    data=data   )
        f.create_dataset('labels',  data=labels )
        f.create_dataset('tlabels', data=tlabels)

data_file.close()
meta_file.close()
