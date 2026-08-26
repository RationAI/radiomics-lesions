import hydra
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@with_cli_args(["+preprocessing=tiling"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    # TODO: Implement tiling logic using ratiopah or rationai.tiling library
    # References: https://github.com/RationAI/ratiopath
    #             https://rationai.gitlab-pages.ics.muni.cz/digital-pathology/libraries/tiling/
    logger.log_artifacts(config.data_path, config.mlflow_artifact_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
