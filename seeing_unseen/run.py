import os

import hydra
from omegaconf import DictConfig

from seeing_unseen.core.base import BaseTrainer
from seeing_unseen.core.registry import registry


@hydra.main(
    config_path=os.path.join(os.getcwd(), "config/baseline/"),
    config_name="clip_unet",
)
def main(cfg: DictConfig):
    trainer: BaseTrainer = registry.get_trainer(cfg.training.trainer)(
        cfg=cfg,
        dataset_dir=cfg.dataset.root_dir,
        checkpoint_dir=cfg.checkpoint_dir,
        log_dir=cfg.log_dir,
    )
    if cfg.run_type == "train":
        trainer.train()
    else:
        trainer.eval(cfg.checkpoint_dir)


if __name__ == "__main__":
    main()
