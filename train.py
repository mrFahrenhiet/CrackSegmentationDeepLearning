import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchcontrib.optim import SWA
from time import time
from tqdm import tqdm
import gc
import argparse
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# local imports
from model.modelMain import Unet
from dataLoader import CrackData
from utils.averageMeter import AverageMeter
from utils.callbacks import CallBacks
from utils.utils import *
from utils.lrSchedular import OneCycleLR
from config import train_loader_config, val_loader_config

RANDOM_STATE = 42


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
    with torch.inference_mode():
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


def getDataLoaders(dfTrain, dfVal, **kwargs):
    dataTrain = CrackData(dfTrain,
                          img_transforms=kwargs['training_data']['transforms'],
                          mask_transform=kwargs['training_data']['transforms'],
                          aux_transforms=None)

    trainLoader = DataLoader(dataTrain,
                             batch_size=kwargs['training_data']['batch_size'],
                             shuffle=kwargs['training_data']['suffle'],
                             pin_memory=torch.cuda.is_available(),
                             num_workers=kwargs['training_data']['num_workers'])

    dataVal = CrackData(dfVal,
                        img_transforms=kwargs['val_data']['transforms'],
                        mask_transform=kwargs['val_data']['transforms'],
                        aux_transforms=None)
    valLoader = DataLoader(dataVal,
                           batch_size=kwargs['val_data']['batch_size'],
                           shuffle=kwargs['val_data']['suffle'],
                           pin_memory=torch.cuda.is_available(),
                           num_workers=kwargs['val_data']['num_workers'])

    return trainLoader, valLoader


def buildModel(encoderBackbone='efficientnet-b2'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet(encoder_name=encoderBackbone)
    model = model.to(device)
    return model


def buildDataset(imgs_path, masks_path):
    data = {
        'images': sorted(glob(imgs_path + "/*.jpg")),
        'masks': sorted(glob(masks_path + "/*.png"))
    }

    # test to see if there are images coresponding to masks
    for img_path, mask_path in zip(data['images'], data['masks']):
        assert img_path[:-4] == mask_path[:-4]

    df = pd.DataFrame(data)
    dfTrain, dfVal = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)
    trainLoader, valLoader = getDataLoaders(dfTrain,
                                            dfVal,
                                            training_data=train_loader_config,
                                            val_data=val_loader_config)

    return trainLoader, valLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_images", type=str, help="Enter path to images folder.")
    parser.add_argument("--path_masks", type=str, help="Enter path to masks folder.")
    parser.add_argument("--out_path", type=str, help="Output path, model saving path.")
    args = parser.parse_args()

    image_path = args.path_images
    masks_path = args.path_masks
    out_path = args.out_path

    trainLoader, valLoader = buildDataset(image_path, masks_path)

    model = buildModel(encoderBackbone='efficientnet-b2')

    lr = 0.09
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    optimizer = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.06)
    schedular = OneCycleLR(optimizer, num_steps=50, lr_range=(1e-5, 0.1), annihilation_frac=0.75)
    criteria = DiceLoss()
    epochs = 50
    accumulation_steps = 4
    best_dice = 0.70
    scaler = GradScaler()
    out_path_model = os.path.join(out_path, "models")
    iteration = 0

    cb = CallBacks(best_dice, out_path_model)

    results = {"train_loss": [], "train_dice": [], "train_iou": [], 'train_acc': [],
               "train_pre": [], "train_rec": [], "train_f1": [],
               "val_loss": [], "val_dice": [], "val_iou": [], "val_acc": [],
               "val_pre": [], "val_rec": [], "val_f1": []}

    save_path = out_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        model_path = out_path_model
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))

    earlyStopEpoch = 10

    try:
        for epoch in range(1, epochs + 1):
            iteration = epoch
            train_logs = train_step(model, optimizer, criteria, trainLoader, accumulation_steps, scaler, epoch, epochs)
            print("\n")
            val_logs = val(model, criteria, valLoader, epoch, epochs)
            print("\n")
            schedular.step()

            results['train_loss'].append(train_logs['loss'].avg)
            results['train_dice'].append(train_logs['dice'].avg)
            results['train_iou'].append(train_logs['iou'].avg)
            results['train_acc'].append(train_logs['acc'].avg)
            results['train_pre'].append(train_logs['precision'].avg)
            results['train_rec'].append(train_logs['recall'].avg)
            results['train_f1'].append(train_logs['f1'].avg)
            results['val_loss'].append(val_logs['loss'].avg)
            results['val_dice'].append(val_logs['dice'].avg)
            results['val_iou'].append(val_logs['iou'].avg)
            results['val_acc'].append(val_logs['acc'].avg)
            results['val_pre'].append(val_logs['precision'].avg)
            results['val_rec'].append(val_logs['recall'].avg)
            results['val_f1'].append(val_logs['f1'].avg)

            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            data_frame.to_csv(f'{save_path}/logs_2.csv', index_label='epoch')

            print("\n")

            cb.saveBestModel(val_logs['dice'].avg, model)
            cb.earlyStoping(val_logs['dice'].avg, earlyStopEpoch)

    except KeyboardInterrupt:
        data_frame = pd.DataFrame(data=results, index=range(1, iteration + 1))
        data_frame.to_csv(f'{save_path}/logs_2.csv', index_label='epoch')
        val_logs = val(model, criteria, valLoader, 1, 1)
        cb.saveBestModel(val_logs['dice'].avg, model)
