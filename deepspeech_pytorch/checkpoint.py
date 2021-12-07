import os
from pathlib import Path

import hydra
from google.cloud import storage
from hydra_configs.pytorch_lightning.callbacks import ModelCheckpointConf
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm


class CheckpointHandler(ModelCheckpoint):

    def __init__(self, cfg: ModelCheckpointConf):
        super().__init__(
            dirpath="checkpoints",
            filename=cfg.filename,
            monitor="wer",
            verbose=cfg.verbose,
            save_last=cfg.save_last,
            save_top_k=cfg.save_top_k,
            save_weights_only=cfg.save_weights_only,
            mode='max',
            period=cfg.period,
            #prefix=cfg.prefix
        )

    def find_latest_checkpoint(self):
        raise NotImplementedError


class FileCheckpointHandler(CheckpointHandler):

    def find_latest_checkpoint(self):
        """
        Finds the latest checkpoint in a folder based on the timestamp of the file.
        If there are no checkpoints, returns None.
        :return: The latest checkpoint path, or None if no checkpoints are found.
        """
        paths = list(Path(self.dirpath).rglob(self.prefix + '*'))
        if paths:
            paths.sort(key=os.path.getctime)
            latest_checkpoint_path = paths[-1]
            return latest_checkpoint_path
        else:
            return None

