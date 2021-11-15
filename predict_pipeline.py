import logging
import click
import torch
import pandas as pd

from src.data.make_dataset import read_pred_data, get_preds_data
from entities.predict_pipeline_params import (
    read_predict_params
)
from src.models.model_utils import (
    load_model,
    predict_class
)
from src.models.model_utils import (
    create_model
)

logger = logging.getLogger("pipeline")


def predict_pipeline(settings):
    """
    Основной пайплайн. Здесь получаем и конвертим данные, затем запускаем предикт
    :param settings:
    :return:
    """
    logger.info("Stage: create model")
    # model, _, _, device = create_model(settings.model_params)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model2, _, _, device = create_model(settings.model_params)
    logger.info("Stage: load model")
    # загрузка модели
    # model = load_model(model, settings.model_path)
    model = load_model(settings.model_path)
    # загрузка данных для прогноза
    data = read_pred_data(settings.predict_img_path)
    # конвертация в датасет для прогноза
    predict_dl = get_preds_data(data, settings.model_params.image_size, settings.predict_img_path)
    # выполняем прогноз
    predictions = predict_class(model, predict_dl, device)
    # сохраняем результат
    predictions.to_csv(settings.output_target_path, index=False)
    logger.info("finish prediction")


@click.command("predict_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path: str):
    """
    Получение конфигов и запуск пайплайна
    :param config_path:
    :return:
    """
    settings = read_predict_params(config_path)
    predict_pipeline(settings)


if __name__ == "__main__":
    predict_pipeline_command()
