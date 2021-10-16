import logging

import click
import pandas as pd

from src.data.make_dataset import read_data
from entities.predict_pipeline_params import (
    PredictPipelineParams,
    read_predict_params
)
from log.utils import set_logging_config
from src.models.model_utils import deserialize


logger = logging.getLogger("inference")


def predict_pipeline(settings: PredictPipelineParams):
    set_logging_config(settings.log_params)

    logger.info("Stage: read data")
    df = read_data(settings.input_data_path)

    logger.info("Stage: load model")
    model = deserialize(settings)["model"]

    predictions = model.predict(df.drop('target', axis=1))

    result = pd.Series(predictions, index=df.index, name="prediction")
    result.to_csv(
        settings.output_target_path,
        index=False
    )
    logger.info("finish prediction")
    return result


@click.command("predict_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path: str):
    settings = read_predict_params(config_path)
    predict_pipeline(settings)


if __name__ == "__main__":
    predict_pipeline_command()