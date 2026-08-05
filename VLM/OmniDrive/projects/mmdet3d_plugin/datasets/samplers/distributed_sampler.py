# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
#  Modified by Shihao Wang
# ---------------------------------------------
import math
import torch
from torch.utils.data import DistributedSampler as _DistributedSampler
from .sampler import SAMPLER


@SAMPLER.register_module()
class DistributedSampler(_DistributedSampler):

    def __init__(self,
                 dataset=None,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0,
                 samples_per_gpu=None):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0
        self.samples_per_gpu = samples_per_gpu
        # StreamPETR keeps temporal memory across iterations and assumes a
        # fixed local batch size. Pad each rank to a per-GPU batch multiple.
        if self.shuffle and self.samples_per_gpu is not None:
            if self.samples_per_gpu <= 0:
                raise ValueError(f"samples_per_gpu must be positive, got {self.samples_per_gpu}")
            self.num_samples = int(math.ceil(self.num_samples / self.samples_per_gpu) * self.samples_per_gpu)
            self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        # in case that indices is shorter than half of total_size
        indices = (indices *
                   math.ceil(self.total_size / len(indices)))[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        per_replicas = self.total_size//self.num_replicas
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        indices = indices[self.rank*per_replicas:(self.rank+1)*per_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
