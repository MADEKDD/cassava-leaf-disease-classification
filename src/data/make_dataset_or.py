import pandas as pd
import logging
import cv2
from torchvision import models as tvmodels
from torch.utils.data import Dataset, DataLoader
from sklearn import model_selection
from typing import Tuple
from entities.split_params import SplitParameters

logger = logging.getLogger("data")


def read_data(path_to_data: str) -> pd.DataFrame:
    logger.debug("start read_data")
    df: pd.DataFrame = pd.read_csv(path_to_data)
    logger.debug("stop read_data")
    return df


def train_val_split(
        df: pd.DataFrame,
        split_params: SplitParameters
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.debug("start train_val_split")
    train_df, valid_df = model_selection.train_test_split(
        df,
        shuffle=split_params.shuffle,
        test_size=split_params.val_size,
        random_state=split_params.random_seed
    )
    logger.info("Train shape: {}".format(train_df.shape))
    logger.info("Valid shape: {}".format(valid_df.shape))
    logger.debug("stop train_val_split")
    return train_df, valid_df