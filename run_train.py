import hydra
from hydra.core.config_store import ConfigStore

#from deepspeech_pytorch.configs.train_config import DeepSpeechConfig, AdamConfig, SGDConfig, BiDirectionalConfig, \
#    UniDirectionalConfig, GCSCheckpointConfig
from src.training import train
from omegaconf.dictconfig import DictConfig

cs = ConfigStore.instance()

@hydra.main(config_path="src/configs", config_name="train_config_waveform_future50.yaml")
def hydra_main(cfg: DictConfig):
    train(cfg=cfg)


if __name__ == '__main__':
    hydra_main()

