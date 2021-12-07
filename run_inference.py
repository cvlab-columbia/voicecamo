import hydra
from hydra.core.config_store import ConfigStore

from deepspeech_pytorch.inference import evaluate
from omegaconf.dictconfig import DictConfig

cs = ConfigStore.instance()
@hydra.main(config_path="deepspeech_pytorch/configs", config_name="train_config_waveform_notfirstlayer_future50.yaml")
def main(cfg: DictConfig):
    evaluate(cfg=cfg)


main()
