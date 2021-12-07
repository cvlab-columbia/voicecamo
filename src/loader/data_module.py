import pytorch_lightning as pl
from hydra.utils import to_absolute_path

from src.loader.data_loader import SpectrogramDataset, DSRandomSampler, AudioDataLoader, \
    DSElasticDistributedSampler
import torch
from torch.utils.data import DataLoader
from omegaconf.dictconfig import DictConfig
import pdb

class DeepSpeechDataModule(pl.LightningDataModule):

    def __init__(self,
                 labels: list,
                 data_cfg: DictConfig,
                 normalize: bool,
                 is_distributed: bool):
        super().__init__()
        self.train_path = to_absolute_path(data_cfg.train_path)
        self.val_path = to_absolute_path(data_cfg.val_path)
        self.test_path = to_absolute_path(data_cfg.test_path)
        self.labels = labels
        self.data_cfg = data_cfg
        self.spect_cfg = data_cfg.spect
        self.aug_cfg = data_cfg.augmentation
        self.normalize = normalize
        self.is_distributed = True

    def collate_fn(self,batch):
        def func(p):
            return p[0].size(1)

        batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
        longest_sample = max(batch, key=func)[0]

        freq_size = longest_sample.size(0)


        minibatch_size = len(batch)
        scalars = torch.zeros(minibatch_size)

        max_seqlength = longest_sample.size(1)

        inputs = torch.zeros(minibatch_size, 2, freq_size, max_seqlength)
        mag_noises = torch.zeros(minibatch_size, 2, freq_size, max_seqlength)
        input_percentages = torch.FloatTensor(minibatch_size)
        target_sizes = torch.IntTensor(minibatch_size)
        targets = []
        for x in range(minibatch_size):
            sample = batch[x]
            real = sample[0]
            imag = sample[1]
            target = sample[2]
            mag_noise = sample[3]
            y=sample[4]
            scalars[x] = torch.max(torch.abs(torch.tensor(y)))
            seq_length = real.size(1)
            inputs[x][0].narrow(1, 0, seq_length).copy_(real)
            inputs[x][1].narrow(1, 0, seq_length).copy_(imag)
            mag_noises[x][0].narrow(1, 0, seq_length).copy_(mag_noise[0])
            mag_noises[x][1].narrow(1, 0, seq_length).copy_(mag_noise[1])
            input_percentages[x] = seq_length / float(max_seqlength)
            target_sizes[x] = len(target)
            targets.extend(target)
        targets = torch.tensor(targets, dtype=torch.long)
        return inputs, targets, mag_noises, input_percentages, target_sizes, scalars

    def train_dataloader(self):
        train_dataset = self._create_dataset(self.train_path)
        """if self.is_distributed:
            train_sampler = DSElasticDistributedSampler(
                dataset=train_dataset,
                batch_size=self.data_cfg.batch_size
            )
        else:
            train_sampler = DSRandomSampler(
                dataset=train_dataset,
                batch_size=self.data_cfg.batch_size
            )"""
        train_loader = DataLoader(
            dataset=train_dataset,
            num_workers=self.data_cfg.num_workers,
            batch_size=self.data_cfg.batch_size,
            #batch_sampler=train_sampler,
            collate_fn=self.collate_fn
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = self._create_dataset(self.val_path)
        val_loader = DataLoader(
            dataset=val_dataset,
            num_workers=self.data_cfg.num_workers,
            batch_size=self.data_cfg.batch_size,
            collate_fn=self.collate_fn,
        )
        return val_loader

    def test_dataloader(self):
        val_dataset = self._create_dataset(self.test_path)
        val_loader = DataLoader(
            dataset=val_dataset,
            num_workers=self.data_cfg.num_workers,
            batch_size=self.data_cfg.batch_size,
            collate_fn=self.collate_fn,
        )
        return val_loader


    def _create_dataset(self, input_path):
        dataset = SpectrogramDataset(
            audio_conf=self.spect_cfg,
            input_path=input_path,
            labels=self.labels,
            normalize=True,
            aug_cfg=self.aug_cfg
        )


        return dataset
