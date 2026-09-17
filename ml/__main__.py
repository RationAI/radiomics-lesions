import hydra
from lightning import seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, ListConfig
from rationai.mlkit import Trainer, autolog


@hydra.main(config_path="../configs", config_name="default", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger) -> None:
    seed_everything(config.seed, workers=True)

    data = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model)
    trainer = hydra.utils.instantiate(config.trainer, _target_=Trainer, logger=logger)

    if isinstance(config.mode, ListConfig):
        for mode in config.mode:
            getattr(trainer, mode)(
                model, datamodule=data, ckpt_path=config.checkpoint.get(mode)
            )
    else:
        getattr(trainer, config.mode)(
            model, datamodule=data, ckpt_path=config.checkpoint
        )


if __name__ == "__main__":
    main()
