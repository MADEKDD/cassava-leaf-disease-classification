import logging
from typing import Dict
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.models.model_utils import val_one_epoch

logger = logging.getLogger("models")




def validate_model(model, optimizer, loss, val_dl, device) -> np.array:
    logger.debug("start predict_model")
    with torch.no_grad():
        acc, metrics = val_one_epoch(model, loss, val_dl, device)
    logger.debug("stop predict_model")
    return acc, metrics


def evaluate_model(predicts: np.ndarray, target: np.array) -> Dict[str, float]:
    logger.debug("start evaluate_model")
    scores = {
        "roc_auc_score": metrics.roc_auc_score(
            y_true=target,
            y_score=predicts
        ),
        "precision_recall": metrics.classification_report(
            y_true=target,
            y_pred=predicts
        )
    }
    logger.debug("stop evaluate_model")
    return scores