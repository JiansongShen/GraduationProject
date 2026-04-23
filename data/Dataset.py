from logging import log, INFO
import SimpleITK as sitk
import torch
from SimpleITK import VectorUInt32

from core.config import DataConfig, AsyncLoadConfig
from data import CachedData
from data.data_preprocesser import NiftiImage, resample_in_memory
from utils.helper import find_all_file_paths_recursively


class Dataset:
    def __init__(self, cfg: DataConfig, load_cfg: AsyncLoadConfig, patch_size: tuple[int, int, int] = (128, 128, 128), device: str = "cuda"):
        self.cfg = cfg
        self.load_cfg = load_cfg
        self.all_file_paths: list[str] = find_all_file_paths_recursively(
            cfg.train_dirs
        )
        self.visited_file_paths: list[str] = []
        self.patch_size = patch_size
        self.device = device
        self.cached_data = CachedData(patch_size)
        self.cur_begin_idx = 0

    def load_data(self) -> tuple[CachedData, int]:
        cached_data: CachedData = self.cached_data
        log(INFO, f"Loading data, with begin index {self.cur_begin_idx}, all {find}")
        max_cache_size: float = self.load_cfg.ram_cache_gb
        begin_idx: int = self.cur_begin_idx
        for begin_idx in range(begin_idx, len(self.all_file_paths)):
            img : NiftiImage = sitk.ReadImage(self.cfg.train_dirs[0])
            img = resample_in_memory(img)
            sizes: VectorUInt32 = img.GetSize()
            loaded_size: float = 0.0
            single_size: float = sizes[0] * sizes[1] * sizes[2] * 4 / 1024 / 1024 / 1024
            if loaded_size + single_size > max_cache_size:
                log(INFO, f"RAM cache size exceeded, stop loading")
                if self.device == "cuda":
                    cached_data.to_cuda()
                return cached_data, self.cur_begin_idx
            else:
                cached_data.add_data(
                    torch.from_numpy(sitk.GetArrayFromImage(resample_in_memory(img, self.patch_size)))
                )

        if self.device == "cuda":
            cached_data.to_cuda()

        return cached_data, begin_idx

    def get_data_and_load_new(self) -> tuple[torch.Tensor, torch.Tensor]:
        res1, res2 : torch.Tensor = self.cached_data.get_data()
        async load_data()
        return res1, res2
