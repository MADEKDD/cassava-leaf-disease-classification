import click
import logging
from pprint import pformat
from log.utils import set_logging_config
from entities.train_pipeline_params import (
    TrainPipelineParams, read_train_parameters
)
from src.data.make_dataset import (
    read_data,
    train_val_split
)
from src.features.build_features import (
    build_transformer,
    create_features,
    create_target
)
from src.models.model_utils import (
    create_model,
    serialize
)
from src.models.predict_model import (
    create_inference_pipeline,
    predict_model,
    evaluate_model
)
from src.models.train_model import (
    train_model
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger("pipeline")


def train_pipeline(settings: TrainPipelineParams):
    set_logging_config(settings.log_params)

    logger.info("Stage: read data")
    data = read_data(settings.input_data_path)

    logger.info("Stage: split data")
    train_df, valid_df = train_val_split(
        data,
        settings.split_params
    )

    logger.info("Stage: create train features")
    transformer = build_transformer(settings.feature_params)

    logger.info("Stage: create target")
    train_target = create_target(train_df, settings.feature_params)

    logger.info("Stage: create train features")
    train_features = create_features(
        transformer,
        train_df.drop(settings.feature_params.target, axis=1, errors='ignore'),
    )

    logger.info("Stage: create model")
    model = create_model(settings.model_params)

    logger.info("Stage: train model")
    train_model(model, train_features, train_target)

    logger.info("Stage: serialize model")
    inference_pipeline = create_inference_pipeline(model, transformer)

    logger.info("Stage: scoring model")
    predicts = predict_model(
        inference_pipeline,
        valid_df.drop(settings.feature_params.target, axis=1, errors='ignore'),
    )

    valid_target = create_target(valid_df, settings.feature_params)
    metrics = evaluate_model(
        predicts,
        valid_target
    )
    logger.info("Metrics: {}".format(pformat(metrics)))
    serialize(
        {
            "model": inference_pipeline,
            "metrics": metrics
        },
        # inference_pipeline,
        settings
    )
    return settings.output_model_path, metrics


@click.command("train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    settings = read_train_parameters(config_path)
    train_pipeline(settings)


if __name__ == "__main__":
    train_pipeline_command()