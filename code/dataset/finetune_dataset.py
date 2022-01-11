from torch.utils.data import Dataset
import torch
import numpy as np
import os
import json
from datetime import datetime
from dataset.data_augmentation import transform


class FinetuneDataset(Dataset):
    def __init__(self, file_path, num_features, patch_size, max_length, norm=None):
        """
        :param file_path: path to the folder of the pre-training dataset
        :param num_features: dimension of each pixel
        :param patch_size: patch size
        :param max_length: padded sequence length
        :param norm: mean and std used to normalize the input reflectance
        """

        self.file_path = file_path
        self.max_length = max_length
        self.dimension = num_features
        self.patch_size = patch_size

        self.FileList = os.listdir(file_path)
        self.TS_num = len(self.FileList)            # number of labeled samples
        self.norm = norm

    def __len__(self):
        return self.TS_num

    def __getitem__(self, item):
        file = self.FileList[item]
        file = os.path.join(self.file_path, file)
        with np.load(file) as sample:

            # class label of this time series
            class_label = sample["class_label"]

            # cloudfree image patch time series
            ts_origin = sample["ts"]    # [seq_Length, band_nums, patch_size, patch_size]
            if self.norm is not None:
                m, s = self.norm
                m = np.expand_dims(m, axis=-1)
                s = np.expand_dims(s, axis=-1)
                shape = ts_origin.shape
                ts_origin = ts_origin.reshape((shape[0], shape[1], -1))
                ts_origin = (ts_origin - m) / s
                ts_origin = ts_origin.reshape(shape)
            else:
                ts_origin = ts_origin / 10000.0

            ts_origin = ts_origin.astype('float')

            ts_origin = transform(ts_origin)

            # length of this time series (varies for each sample)
            ts_length = ts_origin.shape[0]

            # padding time series to the same length
            ts_origin = np.pad(ts_origin, ((0, self.max_length - ts_length), (0, 0), (0, 0), (0, 0)),
                               mode='constant', constant_values=0.0)

            # acquisition dates of this time series
            doy = sample["doy"]     # [seq_Length, ]
            doy = np.pad(doy, (0, self.max_length - ts_length), mode='constant', constant_values=0)

            # mask of valid observations
            bert_mask = np.zeros((self.max_length,), dtype=int)
            bert_mask[:ts_length] = 1

        output = {"bert_input": ts_origin,
                  "bert_mask": bert_mask,
                  "timestamp": doy,
                  "class_label": class_label,
                  }

        return {key: torch.from_numpy(value) for key, value in output.items()}

