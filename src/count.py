import json

import hydra
import torch
from deepspeech_pytorch.loader.data_module_noise import DeepSpeechDataModule
from deepspeech_pytorch.model import DeepSpeechSwap
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf.dictconfig import DictConfig
import os
import pdb
from hydra import utils
torch.multiprocessing.set_sharing_strategy('file_system')

import torch
def train(cfg: DictConfig):
    os.chdir(utils.get_original_cwd())

    percent = cfg.trainer.limit_train_batches

    wandb_path = str(percent) + "_percent_residual_" + str(cfg.residual) + "_power_" + str(cfg.power) + "_lr_" + \
                 str(cfg.optim.learning_rate) + "_batchsize_" + str(cfg.data.batch_size)



    if not os.path.isdir(wandb_path):
        os.mkdir(wandb_path)

    os.chdir(wandb_path)

    seed_everything(cfg.SEED)

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
    model = DeepSpeechSwap(
        wandb = cfg.wandb,
        residual=cfg.residual,
        batchnorm = cfg.batchnorm,
        power=cfg.power,
        labels=labels,
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        precision=cfg.trainer.precision,
        spect_cfg=cfg.data.spect
    )

    model.load_state_dict(torch.load("../librispeech_model_statedict.pth"),strict=False)


    training_load = data_loader.train_dataloader()
    list_of_wer=[]
    list_of_cer=[]
    for batch_idx, batch in enumerate(training_load):
        print(batch_idx, " / ", len(training_load))

        thedict = model.training_step(batch,batch_idx)
        list_of_wer.append(thedict['lev_dist']/thedict['nwords'])
        list_of_cer.append(thedict['cer_dist'] / thedict['nchars'])

    print(cfg.power)
    print("wer", torch.mean(torch.tensor(list_of_wer)))
    print("cer",torch.mean(torch.tensor(list_of_cer)))
    print(cfg.data.spect.power)

        #("../../../librispeech_pretrained_v3.ckpt",strict=False)

def twosec(cfg: DictConfig):
    os.chdir(utils.get_original_cwd())

    percent = cfg.trainer.limit_train_batches

    wandb_path = str(percent) + "_percent_residual_" + str(cfg.residual) + "_power_" + str(cfg.power) + "_lr_" + \
                 str(cfg.optim.learning_rate) + "_batchsize_" + str(cfg.data.batch_size)



    if not os.path.isdir(wandb_path):
        os.mkdir(wandb_path)

    os.chdir(wandb_path)

    seed_everything(cfg.SEED)

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

    training_load = data_loader.train_dataloader()
    list_of_wer=[]
    for batch_idx, batch in enumerate(training_load):
        print(batch_idx, " / ", len(training_load))


        newfilepathdecode = thedict['trans_path'].split(".txt")[0] + "_transafter2sec.txt"
        file = open(newfilepathdecode, "w")
        file.write(str(thedict['decoded_output']))
        file.close()

    print(torch.mean(torch.tensor(list_of_wer)))