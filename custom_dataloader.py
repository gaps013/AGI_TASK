
class InfiniteBatchSampler(Sampler):
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

def collate(batch, samples_per_gpu=1):
    stacked=[]
    for i in range(0, len(batch), samples_per_gpu):
        assert isinstance(batch[i]['img'], torch.Tensor)
        ndim = batch[i]['img'].dim()
        assert ndim > 2
        max_shape = [0 for _ in range(2)]
        for dim in range(1, 3):
            max_shape[dim - 1] = batch[i]['img'].size(-dim)
        for sample in batch[i:i + samples_per_gpu]:
            for dim in range(ndim - 2):
                assert batch[i]['img'].size(dim) == sample['img'].size(dim)
            for dim in range(1, 2+ 1):
                max_shape[dim - 1] = max(max_shape[dim - 1],
                                            sample['img'].size(-dim))
        padded_samples = []
        for sample in batch[i:i + samples_per_gpu]:
            pad = [0 for _ in range(4)]
            for dim in range(1, 3):
                pad[2 * dim -
                    1] = max_shape[dim - 1] - sample['img'].size(-dim)
            padded_samples.append(
                F.pad(
                    sample['img'], pad, value=0))
        stacked.append(default_collate(padded_samples))

        stacked.append(default_collate([sample['gt_bboxes'] for sample in batch[i:i + samples_per_gpu]]))
        stacked.append(default_collate([sample['gt_labels'] for sample in batch[i:i + samples_per_gpu]]))
    return stacked

    infinite_sampler = InfiniteBatchSampler(len(dataset), shuffle=shuffle)
    DataLoader( dataset, batch_size=batch_size, sampler=infinite_sampler, num_workers=num_workers, pin_memory=True,
                    collate_fn=partial(collate, samples_per_gpu=batch_size))
