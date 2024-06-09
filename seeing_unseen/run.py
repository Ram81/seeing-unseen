import os

import hydra
from omegaconf import DictConfig

from seeing_unseen.trainer.trainer import SemanticPlacementTrainer


@hydra.main(
    config_path=os.path.join(os.getcwd(), "config/visual_reasoner/"),
    config_name="main",
)
def main(cfg: DictConfig):
    trainer = SemanticPlacementTrainer(
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
