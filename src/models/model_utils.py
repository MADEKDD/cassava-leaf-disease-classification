import torch
import logging
import pickle as pkl
from typing import Any, Dict
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import resnet
import numpy as np
from sklearn import metrics
from entities.train_pipeline_params import ModelParameters, TrainPipelineParams
from entities.predict_pipeline_params import (
    PredictPipelineParams
)

logger = logging.getLogger("models")


def he_init(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        m.bias.data.fill_(0.01)


def val_one_epoch(model, loss, data_loader, device):
    model.eval()
    
    preds_all = []
    targets_all = []
    loss_sum = 0
    sample_num = 0
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs, targets) in pbar:
        imgs = imgs.to(device).float()
        targets = targets.to(device).long()
        
        preds = model(imgs)
        preds_all += [torch.argmax(preds, 1).detach().cpu().numpy()]
        targets_all += [targets.detach().cpu().numpy()]
        
        cost = loss(preds, targets)
        
        loss_sum += cost.item()*targets.shape[0]
        sample_num += targets.shape[0]
    
    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    
    #return (preds_all==targets_all).mean()
    logger.debug("start evaluate_model")
    scores = {
        "precision_recall": metrics.classification_report(
            y_true=targets_all,
            y_pred=preds_all
        ),
         "accuracy": metrics.accuracy_score(
            y_true=targets_all,
            y_pred=preds_all       
        )
    }
    print('accuracy = {:.4f}'.format((preds_all==targets_all).mean()))
    logger.debug("stop evaluate_model")
    return (preds_all==targets_all).mean(), scores

def train_one_epoch(model, loss, optim, data_loader, device):
    model.train()
    scaler = GradScaler()
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs, targets) in pbar:
        imgs = imgs.to(device).float()
        targets = targets.to(device).long()
        optim.zero_grad()
        
        with autocast():
            preds = model(imgs)
            cost = loss(preds, targets)
        
        scaler.scale(cost).backward()
        scaler.step(optim)
        scaler.update()

def create_model(model_params: ModelParameters):
    logger.debug("start create_model")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cassava_resnet18 = nn.Sequential(
    resnet.resnet18(pretrained = model_params.pretrain),
    nn.Linear(1000, model_params.num_classes)
    )
    cassava_resnet18.apply(he_init)
    model = cassava_resnet18.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    weight = torch.tensor([1087/21397, 2189/21397, 2386/21397,13158/21397, 2577/21397], dtype=torch.float, device='cuda:0')
    loss = nn.CrossEntropyLoss(weight=weight).to(device)
    logger.info(f"Created model: {cassava_resnet18}")
    logger.debug("end create_model")
    return model, optimizer, loss, device


def save_model(model, output_path, model_params):
    torch.save(model, output_path + 'cassnet_{}'.format(model_params.epochs))

def load_model():
    model.load_state_dict(torch.load(filepath))
    model.eval()