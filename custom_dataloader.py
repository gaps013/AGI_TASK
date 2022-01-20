from functools import partial
import itertools
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F

def collate(batch, samples_per_gpu=1):
    """
    Puts each data field into a tensor with outer dimension batch size.
    Extend default_collate to add support
    
    The implementation logic is referred to
    https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/collate.py

    """
    stacked=[]
    for i in range(0, len(batch), samples_per_gpu):
        assert isinstance(batch[i]['img'], torch.Tensor)
        ndim = batch[i]['img'].dim()
        assert ndim > 2
        max_shape = [0 for _ in range(2)]
        for dim in range(1, 2 + 1):
            max_shape[dim - 1] = batch[i]['img'].size(-dim)
        for sample in batch[i:i + samples_per_gpu]:
            for dim in range(ndim - 2):
                assert batch[i]['img'].size(dim) == sample['img'].size(dim)
            for dim in range(1, 2+ 1):
                max_shape[dim - 1] = max(max_shape[dim - 1],
                                            sample['img'].size(-dim))
        padded_samples = []
        for sample in batch[i:i + samples_per_gpu]:
            pad = [0 for _ in range(2 * 2)]
            for dim in range(1, 2 + 1):
                pad[2 * dim -
                    1] = max_shape[dim - 1] - sample['img'].size(-dim)
            padded_samples.append(
                F.pad(
                    sample['img'], pad, value=0))
        stacked.append(default_collate(padded_samples))

        stacked.append(
                    [sample['gt_bboxes'] for sample in batch[i:i + samples_per_gpu]])
        stacked.append(
                    [sample['gt_labels'] for sample in batch[i:i + samples_per_gpu]])
    return stacked

class InfiniteBatchSampler(Sampler):
    """
    Samples mini-batchs infintely and each time yields a mini-batch of size 
    eqivalent to the input batch size.

    The implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset (object): The dataset.
        dataset_size (int): When model is :obj:`DistributedDataParallel`,
            it is the number of training samples on each GPU,
            When model is :obj:`DataParallel`, it is
            `num_gpus * samples_per_gpu`.
            Default : 1.
        world_size (int, optional): Number of processes participating in
            distributed training. Default: None.
        rank (int, optional): Rank of current process. Default: None.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the dataset or not. Default: True.
    """ 

    def __init__(self,
                 dataset_size,
                 world_size=1,
                 rank=0,
                 seed=42,
                 shuffle=True):
        assert dataset_size > 0
        self.rank = rank
        self.world_size = world_size
        self.seed = seed if seed is not None else 42
        self.shuffle = shuffle
        self.size = dataset_size

    def __iter__(self):
        start = self.rank
        yield from itertools.islice(self._infinite_indices(), start, None, self.world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size, generator=g)
            else:
                yield from torch.arange(self.size)

def get_dataloader(dataset, batch_size, num_workers=8, shuffle=True):
    infinite_sampler = InfiniteBatchSampler(len(dataset), shuffle=shuffle)
    return DataLoader( dataset, batch_size=batch_size, sampler=infinite_sampler, num_workers=num_workers, pin_memory=False,
                    collate_fn=partial(collate, samples_per_gpu=batch_size))