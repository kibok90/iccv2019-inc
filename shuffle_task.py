import os
import numpy as np

save_dir = 'split'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# CIFAR-100
num_classes = 100
num_episodes = 10

tasks = []
for seed in range(num_episodes):
    np.random.seed(seed)
    tasks.append(np.random.permutation(num_classes).tolist())
np.save('split/class_order_{}.npy'.format(num_classes), tasks)

# ImageNet
num_classes_all = 1000
num_episodes = 10
num_classes = num_classes_all // num_episodes

np.random.seed(0)
order = np.random.permutation(num_classes_all)
tasks = np.reshape(order, [num_episodes, num_classes]).tolist()
np.save('split/imagenet_split_{}.npy'.format(num_classes), tasks)
