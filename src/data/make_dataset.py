import pandas as pd
import logging
import cv2
import albumentations as A
from torchvision import models as tvmodels
from torch.utils.data import Dataset, DataLoader
from sklearn import model_selection
from typing import Tuple
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger("data")


class GetData(Dataset):
    def __init__(self, df, dirr, label_out=True, transform=None):
        super().__init__()
        self.dirr = dirr
        self.label_out = label_out
        self.transform = transform
        self.df = df.reset_index(drop=True).copy()
        if self.label_out == True:
            self.labels = self.df['label'].values
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index:int):
        img = get_img("{}/{}".format(self.dirr, self.df.loc[index]['image_id']))
        if self.label_out == True:
            target = float(self.labels[index])
        
        img = self.transform(image=img)['image']
            
        if self.label_out:    
            return img, target
        if not self.label_out:
            return img


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb


def get_pred_imgs(pred_img_path):
    """
    функция получения списка картинок из папки
    :param pred_img_path:
    :return:
    """
    from os import listdir
    from os.path import isfile, join
    images = [f for f in listdir(pred_img_path) if isfile(join(pred_img_path, f))]
    return images


def read_pred_data(path_to_data: str) -> pd.DataFrame:
    """
    Формируем датасет
    :param path_to_data:
    :return:
    """
    logger.debug("start read_data")
    # получаем список картинок из папки
    images = get_pred_imgs(path_to_data)
    df = pd.DataFrame(columns=['image_id'])
    df["image_id"] = images
    logger.debug("stop read_data")
    return df


def read_data(path_to_data: str) -> pd.DataFrame:
    logger.debug("start read_data")
    df: pd.DataFrame = pd.read_csv(path_to_data)
    df['label'] = df['label'].astype('string')
    logger.debug("stop read_data")
    return df


def get_preds_data(df: pd.DataFrame, image_size: int, pred_img_path: str,):
    """
    формируем датасет для pytorch
    :param df:
    :param image_size:
    :param pred_img_path:
    :return:
    """
    pred_trans = A.Compose([
        A.CenterCrop(image_size, image_size),
        A.Normalize(image_size, image_size),
        ToTensorV2(),
    ])
    pred_ds = GetData(df, pred_img_path, label_out=False, transform=pred_trans)
    pred_dl = DataLoader(
        pred_ds,
        shuffle=True
    )
    return pred_dl


def train_val_split(
        df: pd.DataFrame,
        train_img_path: str,
        batch_size: int,
        num_workers: int,
        image_size: int
) -> Tuple[DataLoader, DataLoader]:
    logger.debug("start train_val_split")

    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]

    img_size = image_size


    train_df, valid_df = model_selection.train_test_split(
        df,
    )
    train_trans = A.Compose([
        A.RandomResizedCrop(img_size, img_size),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(img_mean, img_std),
        ToTensorV2(),
    ])

    val_trans = A.Compose([
        A.CenterCrop(img_size, img_size),
        A.Normalize(img_mean, img_std),
        ToTensorV2(),
    ])

    test_trans = A.Compose([
        A.CenterCrop(img_size, img_size),
        A.Normalize(img_mean, img_std),
        ToTensorV2(),
    ])
        
    train_ds = GetData(train_df, train_img_path, label_out=True, transform=train_trans)
    valid_ds = GetData(valid_df, train_img_path, label_out=True, transform=val_trans)
    
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,   
    )
    val_dl = DataLoader(
        valid_ds, 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    return train_dl, val_dl


    train_df, valid_df = model_selection.train_test_split(
        df,
    )
    logger.info("Train shape: {}".format(train_df.shape))
    logger.info("Valid shape: {}".format(valid_df.shape))
    logger.debug("stop train_val_split")
    return train_df, valid_df