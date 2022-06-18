import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from time import time
from tqdm import tqdm
import gc
import argparse

# local imports
from model.modelMain import Unet
from dataLoader import CrackData
from utils.averageMeter import AverageMeter
from utils.callbacks import CallBacks
from utils.utils import *
from utils.lrSchedular import OneCycleLR


def init_log():
    log = {
        'loss': AverageMeter(),
        'time': AverageMeter(),
        'iou': AverageMeter(),
        'dice': AverageMeter(),
        'acc': AverageMeter(),
        'precision': AverageMeter(),
        'recall': AverageMeter(),
        'f1': AverageMeter()
    }
    return log


def train_step(model, optim, criteria, loader, accumulation_steps, scaler, epoch, max_epochs):
    model.train()
    train_logs = init_log()
    bar = tqdm(loader, dynamic_ncols=True)
    torch.cuda.empty_cache()
    start = time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.enable_grad():
        for idx, data in enumerate(bar):
            imgs, masks = data
            imgs, masks = imgs.to(device), masks.to(device)

            with autocast():
                output = model(imgs)
                output = output.squeeze(1)
                op_preds = torch.sigmoid(output)
                masks = masks.squeeze(1)
                # loss = criteria(op_preds, masks)
                loss = criteria(op_preds, masks) / accumulation_steps

            batch_size = imgs.size(0)

            scaler.scale(loss).backward()

            if ((idx + 1) % accumulation_steps == 0) or (idx + 1 == len(loader)):
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

            train_logs['loss'].update(loss.item(), batch_size)
            train_logs['time'].update(time() - start)
            train_logs['dice'].update(compute_dice2(op_preds, masks).item(), batch_size)
            train_logs['iou'].update(get_IoU(op_preds, masks).item(), batch_size)
            train_logs['acc'].update(accuracy(op_preds, masks).item(), batch_size)
            p, r, f = precision_recall_f1(op_preds, masks)
            train_logs['precision'].update(p.item(), batch_size)
            train_logs['recall'].update(r.item(), batch_size)
            train_logs['f1'].update(f.item(), batch_size)

            bar.set_description(f"Training Epoch: [{epoch}/{max_epochs}] Loss: {train_logs['loss'].avg}"
                                f" Dice: {train_logs['dice'].avg} IoU: {train_logs['iou'].avg}"
                                f" Accuracy: {train_logs['acc'].avg} Precision: {train_logs['precision'].avg}"
                                f" Recall: {train_logs['recall'].avg} F1: {train_logs['f1'].avg}")
            del imgs
            del masks
            gc.collect()

    return train_logs


def val(model, criteria, loader, epoch, epochs, split='Validation'):
    model.eval()
    val_logs = init_log()
    bar = tqdm(loader, dynamic_ncols=True)
    start = time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for idx, data in enumerate(bar):
            imgs, masks = data
            imgs, masks = imgs.to(device), masks.to(device)

            output = model(imgs)
            output = output.squeeze(1)
            op_preds = torch.sigmoid(output)
            masks = masks.squeeze(1)
            loss = criteria(op_preds, masks)

            batch_size = imgs.size(0)
            val_logs['loss'].update(loss.item(), batch_size)
            val_logs['time'].update(time() - start)
            val_logs['dice'].update(compute_dice2(op_preds, masks).item(), batch_size)
            val_logs['iou'].update(get_IoU(op_preds, masks).item(), batch_size)
            val_logs['acc'].update(accuracy(op_preds, masks).item(), batch_size)
            p, r, f = precision_recall_f1(op_preds, masks)
            val_logs['precision'].update(p.item(), batch_size)
            val_logs['recall'].update(r.item(), batch_size)
            val_logs['f1'].update(f.item(), batch_size)

            bar.set_description(f"{split} Epoch: [{epoch}/{epochs}] Loss: {val_logs['loss'].avg}"
                                f" Dice: {val_logs['dice'].avg} IoU: {val_logs['iou'].avg}"
                                f" Accuracy: {val_logs['acc'].avg} Precision: {val_logs['precision'].avg}"
                                f" Recall: {val_logs['recall'].avg} F1: {val_logs['f1'].avg}")

    return val_logs


def getDataLoaders(dfTrain, dfVal, dfTest, **kwargs):
    data_train = CrackData(dfTrain,
                           img_transforms=kwargs['training_data']['transforms'],
                           mask_transform=kwargs['training_data']['transforms'],
                           aux_transforms=None)

    train_loader = DataLoader(data_train,
                              batch_size=kwargs['training_data']['batch_size'],
                              shuffle=kwargs['training_data']['suffle'],
                              pin_memory=torch.cuda.is_available(),
                              num_workers=kwargs['training_data']['num_workers'])

    data_val = CrackData(dfVal,
                         img_transforms=kwargs['val_data']['transforms'],
                         mask_transform=kwargs['val_data']['transforms'],
                         aux_transforms=None)
    val_loader = DataLoader(data_val,
                            batch_size=kwargs['val_data']['batch_size'],
                            shuffle=kwargs['val_data']['suffle'],
                            pin_memory=torch.cuda.is_available(),
                            num_workers=kwargs['val_data']['num_workers'])

    data_test = CrackData(dfTest,
                          img_transforms=kwargs['test_data']['transforms'],
                          mask_transform=kwargs['test_data']['transforms'],
                          aux_transforms=None)
    test_loader = DataLoader(data_test,
                             batch_size=kwargs['test_data']['batch_size'],
                             shuffle=kwargs['test_data']['suffle'],
                             pin_memory=torch.cuda.is_available(),
                             num_workers=kwargs['test_data']['num_workers'])

    return train_loader, val_loader, test_loader


def buildModel(encoderBackbone='efficientnet-b2'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet(encoder_name=encoderBackbone)
    model = model.to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
