from unittest.mock import patch

import torch
from torch import Tensor


class CachedData:
    """cached data for async loading"""

    def __init__(self, patch_size: tuple[int, int, int] = (128, 128, 128),
                 data_train: torch.Tensor = None, data_label: torch.Tensor = None):
        self.data_train = data_train
        self.data_label = data_label
        self.patch_size = patch_size

    def to_cuda(self):
        """move data to cuda"""
        self.data_train = self.data_train.cuda()
        self.data_label = self.data_label.cuda()

    def _split_to_patches(self, data: torch.Tensor) -> torch.Tensor:
        """patch shape: batch * channel * 3D"""
        base_shape: list = [1, 1]
        base_shape.extend(self.patch_size)
        res: torch.Tensor = torch.tensor(self.patch_size)
        for i in range(data.shape[0], self.patch_size[0]):
            for j in range(data.shape[1], self.patch_size[1]):
                for k in range(data.shape[2], self.patch_size[2]):
                    t = torch.zeros(base_shape)
                    if i + self.patch_size[0] > data.shape[0] or j + self.patch_size[1] > data.shape[1] or k + \
                            self.patch_size[2] > data.shape[2]:
                        t[:,
                        :,
                        0: data.shape[0] - i
                        if i + self.patch_size[0] > data.shape[0]
                        else self.patch_size[0],
                        0: data.shape[1] - j
                        if j + self.patch_size[1] > data.shape[1]
                        else self.patch_size[1],
                        0: data.shape[2] - k
                        if k + self.patch_size[2] > data.shape[2]
                        else self.patch_size[2],
                        ] = data[
                            i:data.shape[0] if i + self.patch_size[0] > data.shape[0] else i + self.patch_size[0],
                            j:data.shape[1] if j + self.patch_size[1] > data.shape[1] else j + self.patch_size[1],
                            k:data.shape[2] if k + self.patch_size[2] > data.shape[2] else k + self.patch_size[2]]
                    else:
                        t[:, :, i:i + self.patch_size[0], j:j + self.patch_size[1], k:k + self.patch_size[2]] = data[
                            i:i + self.patch_size[0], j:j + self.patch_size[1], k:k + self.patch_size[2]]
                    res = torch.cat((res, t), dim=0)
        return res

    def add_train_data(self, data: torch.Tensor):
        """add data to cached data"""
        if self.data_train is None:
            self.data_train = self._split_to_patches(data)
        else:
            self.data_train = torch.cat((self.data_train, self._split_to_patches(data)), dim=0)

    def add_label_data(self, data: torch.Tensor):
        """add label to cached data"""
        if self.data_label is None:
            self.data_label = self._split_to_patches(data)
        else:
            self.data_label = torch.cat((self.data_label, self._split_to_patches(data)), dim=0)

    def get_data(self) -> tuple[Tensor | None, Tensor | None]:
        """get data"""
        return self.data_train, self.data_label
