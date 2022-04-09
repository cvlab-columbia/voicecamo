import hydra
from hydra.core.config_store import ConfigStore
from src.inference import evaluate
from omegaconf.dictconfig import DictConfig

cs = ConfigStore.instance()
@hydra.main(config_path="src/configs", config_name="train_config_future50.yaml")
def main(cfg: DictConfig):
    evaluate(cfg=cfg)

main()
