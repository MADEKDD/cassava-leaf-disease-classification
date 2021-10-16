import numpy as np
import logging
import torch.nn as nn
import torch
from torchvision.models import resnet
from src.models.model_utils import train_one_epoch
from src.models.predict_model import validate_model



logger = logging.getLogger("models")


def train_model(model, optimizer, loss, device, train_dl, valid_dl, model_params):
    logger.debug("start train_model")
    for epochs in range(model_params.epochs):
        train_one_epoch(model, loss, optimizer, train_dl, device)
        acc, metrics = validate_model(model, optimizer, loss, valid_dl, device)
    return model
