import json
from deepspeech_pytorch.loader.data_module import DeepSpeechDataModule
from deepspeech_pytorch.model_copy import DeepSpeech, AudioVisualNet, JointModel
from hydra.utils import to_absolute_path
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf.dictconfig import DictConfig
import os
from hydra import utils
from pytorch_lightning.callbacks import LearningRateMonitor
import torch

torch.autograd.set_detect_anomaly(True)

def train(cfg: DictConfig):

    os.chdir(utils.get_original_cwd())

    percent = cfg.data.spect.size

    waveform=cfg.waveform
    firstlayer=cfg.firstlayer
    capped=cfg.capped
    power= cfg.power
    power_penalization=cfg.powerpenalization
    future = cfg.future

    if capped:
        wandb_path = "future50_clean_notdenoiser_waveform_scalar_" + str(waveform) + "_future_" + str(future) + "_future_amt" + str(cfg.future_amt) +  "_firstlayer_" + str(firstlayer) + "_capped_" + str(capped)  + "_power_" + str(power) + "_lr_" + \
                     str(cfg.optim.learning_rate)
    else:
        wandb_path = "test_waveform_scalar_" + str(waveform) + "_future_" + str(future) + "_future_amt" + str(cfg.future_amt) + "_firstlayer_" + str(firstlayer) + "_capped_" + str(
            capped) + "_powerpen_" + str(power_penalization) + "_lr_" + \
                     str(cfg.optim.learning_rate)


    if not os.path.isdir(wandb_path):
        os.mkdir(wandb_path)

    os.chdir(wandb_path)
    seed_everything(cfg.SEED)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    with open(to_absolute_path(cfg.data.labels_path)) as label_file:
        labels = json.load(label_file)

    if cfg.trainer.accelerator == "ddp":
        is_distributed=True
    else:
        is_distributed=False

    data_loader = DeepSpeechDataModule(
        labels=labels,
        data_cfg=cfg.data,
        normalize=True,
        is_distributed=is_distributed
    )

    model = DeepSpeech(
        wandb=cfg.wandb,
        future=cfg.future,
        future_amt=cfg.future_amt,
        residual=cfg.residual,
        batchnorm=cfg.batchnorm,
        waveform=cfg.waveform,
        firstlayer=cfg.firstlayer,
        capped=cfg.capped,
        inputreal=cfg.inputreal,
        power=cfg.power,
        labels=labels,
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        precision=cfg.trainer.precision,
        spect_cfg=cfg.data.spect
    )


    model.load_state_dict(torch.load("../librispeech_model_statedict.pth"),strict=False)

    if cfg.wandb:
        logger = pl_loggers.WandbLogger(name=wandb_path, save_dir="", project="noise")


    checkpoint_file = "{epoch:02d}-step{step:.2f}"

    checkpoint_callback = ModelCheckpoint(filename=checkpoint_file,dirpath="check", every_n_train_steps=50, save_top_k=-1)

    if cfg.wandb:
        trainer = Trainer(callbacks=[lr_monitor,checkpoint_callback],logger = logger,
            **cfg.trainer)
    else:
        trainer = Trainer(callbacks=[lr_monitor,checkpoint_callback],
                          **cfg.trainer)

    trainer.fit(model,data_loader.train_dataloader(),data_loader.val_dataloader())