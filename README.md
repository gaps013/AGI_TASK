# AGI_TASK

This repository contains the code for Applied AGI interview task.

Custom dataset contains the code for creating a dataset class which can handle dataset containing images and annotations.
It also contains code for transforming the dataset.

Custom dataloader contains the code for creating a dataloader class than can load mini batches indefinately. This helps create a continuous stream of data. It uses a custom collate function and a custom sampler.
