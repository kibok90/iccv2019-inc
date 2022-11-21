from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch
import torch.utils.data as data
from .utils import download_url, check_integrity
import h5py

class iCIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, dataset, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 slabels=None, tasks=None, exr=1., seed=-1):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        self.tasks = tasks
        self.exr = exr
        self.seed = seed
        self.t = -1

        if slabels is not None:
            self.CUR, self.PRE, self.EXT, self.OOD = slabels
        else:
            self.CUR, self.PRE, self.EXT, self.OOD = 1, 2, 4, 8

        # targets as numpy.array
        self.targets = np.array(self.targets)
        self.srcs    = np.full_like(self.targets, self.CUR, dtype=np.uint8)

        # split dataset
        if tasks is None:
            self.archive = [(self.data.copy(), self.targets.copy())]
        else:
            self.archive = []
            for task in tasks:
                locs = np.isin(self.targets, task).nonzero()[0]
                self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))

        if self.train:
            self.coreset = (np.zeros(0, dtype=self.data.dtype), np.zeros(0, dtype=self.targets.dtype))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, src) where target is index of the target class and src is the type of the source.
        """
        img, target, src = self.data[index], self.targets[index], self.srcs[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, src

    def load_dataset(self, prev, t, train=True):
        if train:
            self.data, self.targets = self.archive[t]
        else:
            self.data    = np.concatenate([self.archive[s][0] for s in range(t+1)], axis=0)
            self.targets = np.concatenate([self.archive[s][1] for s in range(t+1)], axis=0)
        self.srcs = np.full_like(self.targets, self.CUR, dtype=np.uint8)
        self.srcs[np.isin(self.targets, prev)] |= self.PRE
        self.t = t

    def append_coreset(self, only=False, interp=False):
        if self.train and (len(self.coreset[0]) > 0):
            if only:
                self.data, self.targets = self.coreset
                self.srcs = np.full_like(self.targets, self.PRE, dtype=np.uint8)
                self.srcs[np.isin(self.targets, self.tasks[self.t])] |= self.CUR
            else:
                srcs = np.full_like(self.coreset[1], self.PRE, dtype=np.uint8)
                srcs[np.isin(self.coreset[1], self.tasks[self.t])] |= self.CUR
                self.data = np.concatenate([self.data, self.coreset[0]], axis=0)
                self.targets = np.concatenate([self.targets, self.coreset[1]], axis=0)
                self.srcs = np.concatenate([self.srcs, srcs], axis=0)

    def append_ex(self, ex_dataset, keep_label=False):
        if len(ex_dataset) > 0:
            self.data = np.concatenate([self.data, ex_dataset.data], axis=0)
            if keep_label: self.targets = np.concatenate([self.targets, ex_dataset.targets], axis=0)
            else:          self.targets = np.concatenate([self.targets, np.full_like(ex_dataset.targets, -1)], axis=0)
            self.srcs = np.concatenate([self.srcs, np.full_like(ex_dataset.targets, self.EXT, dtype=np.uint8)], axis=0)
        if hasattr(ex_dataset, 'ood') and (ex_dataset.ood is not None):
            self.data = np.concatenate([self.data, ex_dataset.ood], axis=0)
            self.targets = np.concatenate([self.targets, np.full(len(ex_dataset.ood), -2)], axis=0)
            self.srcs = np.concatenate([self.srcs, np.full(len(ex_dataset.ood), self.OOD, dtype=np.uint8)], axis=0)

    def remove_ex(self):
        locs = (self.srcs & (self.CUR | self.PRE)) > 0
        self.data = self.data[locs]
        self.targets = self.targets[locs]
        self.srcs = self.srcs[locs]

    def get_stats(self, prev, seen, d_only_in_t=False, p_knows_t=True, dwex=False, dwood=False):
        # tasks, t = self.tasks, self.t
        tasks = self.tasks[:self.t+1]

        locs_seen = (self.srcs & (self.CUR | self.PRE)) > 0
        num_seen = locs_seen.astype(int).sum()
        targets_seen = self.targets[locs_seen]
        if p_knows_t:
            targets_cur = targets_seen
            prev = seen
        else:
            locs_cur = (self.srcs & self.CUR) > 0
            targets_cur = self.targets[locs_cur]

        locs_ext = (self.srcs & self.EXT) > 0
        locs_ood = (self.targets & self.OOD) > 0
        num_ext = locs_ext.astype(int).sum()
        num_ood = locs_ood.astype(int).sum()

        stats_seen = np.zeros(len(seen), dtype=np.float32)
        for i, k in enumerate(seen):
            stats_seen[i] = (targets_seen == k).astype(int).sum()
        if dwex and (num_ext > 0): # external data are expected to be in previous tasks
            for i, k in enumerate(seen):
                if k in prev:
                    stats_seen[i] += num_ext * self.exr / len(prev)
        if dwood and (num_ood > 0): # ood are expected to be in any tasks
            stats_seen += num_ood * self.exr / len(seen)

        stats_prev = np.zeros(len(prev), dtype=np.float32)
        for i, k in enumerate(prev):
            stats_prev[i] = (targets_seen == k).astype(int).sum()
        if dwex: # labeled but not in prev
            stats_prev += (num_seen - stats_prev.sum()) / len(prev)
        if dwex and (num_ext > 0):
            stats_prev += num_ext * self.exr / len(prev)
        if dwood and (num_ood > 0):
            stats_prev += num_ood * self.exr / len(prev)

        stats_local = [np.zeros(len(task), dtype=np.float32) for task in tasks]
        for s, task in enumerate(tasks):
            for i, k in enumerate(task):
                stats_local[s][i] = (targets_seen == k).astype(int).sum()
            if dwex: # labeled but not in s
                stats_local[s] += (num_seen - stats_local[s].sum()) / len(task)
        if dwex and (num_ext > 0):
            for s, task in enumerate(tasks):
                stats_local[s] += num_ext * self.exr / len(task)
        if dwood and (num_ood > 0):
            for s, task in enumerate(tasks):
                stats_local[s] += num_ood * self.exr / len(task)

        if (not p_knows_t) and d_only_in_t:
            task = tasks[-1]
            stats_cur = np.zeros(len(task), dtype=np.float32)
            for i, k in enumerate(task):
                stats_cur[i] = (targets_cur == k).astype(int).sum()
            if dwex: # labeled but not in cur
                stats_cur += (len(targets_d) - stats_cur.sum()) / len(task)
            if dwex and (num_ext > 0):
                stats_cur += num_ext * self.exr / len(task)
            if dwood and (num_ood > 0):
                stats_cur += num_ood * self.exr / len(task)
            stats_local[-1] = stats_cur

        return stats_seen, stats_prev, stats_local

    def update_coreset(self, coreset_size, seen):
        num_data_per = coreset_size // len(seen)
        remainder = coreset_size % len(seen)
        data = []
        targets = []

        # random coreset management; latest classes take memory remainder
        # coreset selection without affecting RNG state
        state = np.random.get_state()
        np.random.seed(self.seed*10000+self.t)
        for k in reversed(seen):
            locs = (self.targets == k).nonzero()[0]
            if (remainder > 0) and (len(locs) > num_data_per):
                num_data_k = num_data_per + 1
                remainder -= 1
            else:
                num_data_k = min(len(locs), num_data_per)
            locs_chosen = locs[np.random.choice(len(locs), num_data_k, replace=False)]
            data.append(self.data[locs_chosen])
            targets.append(self.targets[locs_chosen])
        self.coreset = (np.concatenate(list(reversed(data)), axis=0), np.concatenate(list(reversed(targets)), axis=0))
        np.random.set_state(state)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class iCIFAR100(iCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class H5Dataset(iCIFAR10):
    def __init__(self, root, dataset, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 slabels=None, tasks=None, exr=1., seed=-1,
                 etype='none', num_ex=0, load=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.tasks = tasks
        self.exr = exr
        self.seed = seed
        self.t = -1

        if slabels is not None:
            self.CUR, self.PRE, self.EXT, self.OOD = slabels
        else:
            self.CUR, self.PRE, self.EXT, self.OOD = 1, 2, 4, 8

        self.dataset = dataset
        self.seed = seed
        self.etype = etype
        self.num_ex = num_ex

        if etype == 'none' or load:
            self.load_from_file(dataset=dataset, seed=seed, etype=etype, num_ex=num_ex)

            # training classes for imagenet
            if (self.dataset == 'imagenet') and (self.etype == 'none'):
                classes = np.load('split/imagenet_split_100.npy')[seed]
                locs_in  = np.isin(self.targets, classes)
                label_map = np.full(1000, -1)
                for i, j in enumerate(classes): label_map[j] = i

                self.data, self.targets = self.data[locs_in], label_map[self.targets[locs_in]]
                self.srcs = np.full_like(self.targets, self.CUR, dtype=np.uint8)
                self.label_map = label_map

            if self.etype == 'none':
                # split dataset
                if tasks is None:
                    self.archive = [(self.data.copy(), self.targets.copy())]
                else:
                    self.archive = []
                    for task in tasks:
                        locs = np.isin(self.targets, task).nonzero()[0]
                        self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))
            elif self.etype == 'all':
                self.archive = [(self.data, self.targets)]
                self.data    = np.zeros(0, dtype=self.archive[0][0].dtype)
                self.targets = np.zeros(0, dtype=self.archive[0][1].dtype)

            if self.train:
                self.coreset = (np.zeros(0, dtype=self.data.dtype), np.zeros(0, dtype=self.targets.dtype))

    def load_from_file(self, dataset=None, seed=None, etype=None, num_ex=None, load_index=False):
        if dataset is not None:
            self.dataset = dataset
        if seed is not None:
            self.seed = seed
        if etype is not None:
            self.etype = etype
        if num_ex is not None:
            self.num_ex = num_ex
        if load_index:
            dataset_name = self.dataset + '_locs'
            root = self.dataset
            fmt = 'npy'
        else:
            dataset_name = self.dataset
            root = self.root
            fmt = 'h5'

        if self.etype == 'none':
            if self.train:
                num_data_per = 500
                filename = os.path.join(root, '{}_train_{:d}.{}'.format(dataset_name, num_data_per, fmt))
            else:
                num_data_per = 100
                filename = os.path.join(root, '{}_test_{:d}.{}'.format(dataset_name, num_data_per, fmt))
        elif self.etype == 'all':
            filename = os.path.join(root, '{}.{}'.format(dataset_name, fmt))
        else:
            filename = os.path.join(root, '{}_{}_{:d}_{:d}.{}' \
                                          .format(dataset_name, self.etype, self.num_ex, self.seed, fmt))

        if os.path.isfile(filename):
            print('load {}'.format(filename))
            if load_index:
                indexes      = np.load(filename)
                self.data    = self.archive[0][0][indexes]
                self.targets = self.archive[0][1][indexes]
            else:
                with h5py.File(filename, 'r') as f:
                    self.data    = f['data'][:]
                    self.targets = f['labels'][:]
            self.srcs = np.full_like(self.targets, self.CUR, dtype=np.uint8)
        else:
            raise ValueError('{} not found'.format(filename))
