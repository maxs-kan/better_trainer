from __future__ import annotations

import json
import math
import os
import zipfile
from pathlib import Path

import torch
from lightning import LightningDataModule
from safetensors.torch import load
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader as DataLoader


def get_data_enties(index_file) :
    index_file = Path(index_file)
    data = []
    for item in json.loads(index_file.read_text()):
        path, member = (Path(item["file_path"]), item["member"])
        data.append((path if path.is_absolute() else index_file.parent / path, member))
    return data


class PromptCacheDataset(Dataset):
    def __init__(self, index_file: str | Path, limit: int | None = None):
        self.data = get_data_enties(index_file)[:limit]
        if not self.data:
            raise ValueError(f"Empty prompt cache index {index_file}")
        self._files: dict[Path, zipfile.ZipFile] = {}
        self._pid = os.getpid()

    def __getstate__(self) -> dict:
        # Open zip handles are neither picklable nor safe to share across forked workers.
        return {**self.__dict__, "_files": {}}

    def _load_zip(self, path: Path) -> zipfile.ZipFile:
        if self._pid != os.getpid():
            self._files, self._pid = {}, os.getpid()
        if path not in self._files:
            self._files[path] = zipfile.ZipFile(path)
        return self._files[path]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        path, member = self.data[index]
        embeding = load(self._load_zip(path).read(member))
        return {
            "prompt_embeds": embeding["prompt_embeds"],
            "text_token_tags": embeding["text_token_tags"],
            "index": index,
        }


def collate_single(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError(f"PDD uses batch_size=1, got {len(batch)}")
    item = batch[0]
    return {
        "prompt_embeds": item["prompt_embeds"][None],
        "text_token_tags": item["text_token_tags"],
        "index": torch.tensor(item["index"]),
    }


class TrajectorySampler(DistributedSampler):
    """Rank-sharded, shuffled prompt order where each prompt is yielded `n_repeats` times in a row.
    trainer must use `use_distributed_sampler: false`.
    """
    def __init__(self, num_repeats, *args, **kwargs):
        self.num_repeats = num_repeats
        super().__init__(*args, **kwargs)

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        if len(indices) != self.total_size:
            raise AssertionError(
                f"Number of indices ({len(indices)}) does not match total_size ({self.total_size})"
            )

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise AssertionError(
                f"Number of subsampled indices ({len(indices)}) does not match num_samples ({self.num_samples})"
            )

        for index in indices:
            for _ in range(self.num_repeats):
                yield index
    
    def __len__(self) -> int:
        return self.num_samples * self.num_repeats


class PDDPromptDataModule(LightningDataModule):
    def __init__(
        self,
        train_index: str | Path,
        val_index: str | Path | None = None,
        block_steps: int = 8,
        max_val_prompts: int | None = 8,
        num_workers: int = 2,
        pin_memory: bool = True,
        seed: int = 42,
        prefetch_factor: int = 2,
    ):

        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_dataset: PromptCacheDataset | None = None
        self.val_dataset: PromptCacheDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.trainer is not None and self.trainer._accelerator_connector.use_distributed_sampler:
            raise RuntimeError("PDDPromptDataModule shards prompts itself: set trainer.use_distributed_sampler=false")
        if stage in ("fit", None):
            self.train_dataset = PromptCacheDataset(self.hparams.train_index)
        if stage in ("fit", "validate", None) and self.hparams.val_index:
            self.val_dataset = PromptCacheDataset(self.hparams.val_index, self.hparams.max_val_prompts)

    def _world(self) -> tuple[int, int]:
        if self.trainer is None:
            return 1, 0
        return self.trainer.world_size, self.trainer.global_rank

    def train_dataloader(self) -> DataLoader:
        world_size, rank = self._world()
        sampler = TrajectorySampler(
            num_repeats=self.hparams.block_steps,
            dataset=self.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=self.hparams.seed,
            drop_last=True,
        )
        sampler.set_epoch(self.trainer.current_epoch if self.trainer is not None else 0)

        return DataLoader(
            self.train_dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            collate_fn=collate_single,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader | list:
        if self.val_dataset is None:
            return None
        world_size, rank = self._world()
        sampler = DistributedSampler(self.val_dataset, world_size, rank, shuffle=False, drop_last=False)
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            collate_fn=collate_single,
            drop_last=False,
        )
