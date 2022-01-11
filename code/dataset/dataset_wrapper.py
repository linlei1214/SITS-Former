import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataset.pretrain_dataset import PretrainDataset


class DataSetWrapper(object):

    def __init__(self, data_path, batch_size, valid_size, num_features, patch_size,
                 max_length, mask_rate, num_workers):

        self.data_path = data_path
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.num_features = num_features
        self.patch_size = patch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.mask_rate = mask_rate

    def get_data_loaders(self):
        dataset = PretrainDataset(self.data_path, self.num_features, self.patch_size, self.max_length,
                                  mask_rate=self.mask_rate)
        train_loader, valid_loader = self.get_train_validation_data_loaders(dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, dataset):
        # obtain training indices that will be used for validation
        num_train = len(dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=False, pin_memory=True)

        valid_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=False, pin_memory=True)

        return train_loader, valid_loader
