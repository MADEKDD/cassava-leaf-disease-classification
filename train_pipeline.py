import click
import logging
from pprint import pformat
from logs.utils import set_logging_config
from entities.train_pipeline_params import (
    TrainPipelineParams, read_train_parameters
)
from src.data.make_dataset import (
    read_data,
    train_val_split
)

from src.models.model_utils import (
    create_model
)
from src.models.train_model import (
    train_model
)
from src.models.predict_model import (
    validate_model
)
from src.models.model_utils import (
    save_model
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger("pipeline")


def train_pipeline(settings: TrainPipelineParams):
    set_logging_config(settings.log_params)

    logger.info("Stage: read data")
    data = read_data(settings.train_csv_data_path)

    logger.info("Stage: split data")
    train_df, valid_df = train_val_split(
        data,
        settings.train_img_path, 
        settings.model_params.batch_size, 
        settings.model_params.num_workers,
        settings.model_params.image_size
    )

    logger.info("Stage: create model")
    model, optimizer, loss, device  = create_model(settings.model_params)
    logger.info("Stage: train model")
    model = train_model(
        model, 
        optimizer, 
        loss, 
        device, 
        train_df,
        valid_df,
        settings.model_params
    )
    logger.info("Start model save")
    save_model(model, settings.output_model_path, settings.model_params)
    logger.info("Stage: scoring model")
    acc, metrics = validate_model(model, optimizer, loss, valid_df, device
    )
    logger.info("Metrics: {}".format(pformat(metrics)))
    return settings.output_model_path, metrics


@click.command("train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    settings = read_train_parameters(config_path)
    train_pipeline(settings)


if __name__ == "__main__":
    train_pipeline_command()