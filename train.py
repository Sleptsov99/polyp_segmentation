import torch
import torch.nn as nn
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import build_model

def dice_coefficient(pred,target,threshold=0.5):
    pred=(torch.sigmoid(pred)>threshold).float()
    
    intersection=(pred*target).sum()
    
    return((2.0*intersection)  /(pred.sum()+target.sum()+1e-8))

def train_one_epoch(model,loader,optimizer, criterion):
    model.train()
    total_loss=0
    total_dice=0
    for images,masks in tqdm(loader,desc="Train"):
        images=images.to(config.DEVICE)
        masks=masks.to(config.DEVICE)
        
        optimizer.zero_grad()
        pred=model(images)
        loss=criterion(pred,masks)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        total_dice+=dice_coefficient(pred,masks).item()
    n=len(loader)
    return total_loss/n, total_dice/n

def validate(model,loader,criterion):
    model.eval()
    total_loss=0
    total_dice=0
    with torch.no_grad():
        for images, masks in tqdm(loader,desc="Val"):
            images=images.to(config.DEVICE)
            masks=masks.to(config.DEVICE)
            
            preds=model(images)
            loss=criterion(preds,masks)
            
            total_loss+=loss.item()
            total_dice+=dice_coefficient(preds,masks).item()
    n=len(loader)
    return total_loss/n, total_dice/n

def train():
    train_loader, val_loader=get_dataloaders()
    model=build_model()
    optimizer=torch.optim.Adam(model.parameters(),lr=config.LEARNING_RATE)
    criterion=nn.BCEWithLogitsLoss()
    best_dice=0.0
    for epoch in range(1, config.NUM_EPOCHS+1):
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_dice   = validate(model, val_loader, criterion)

        print(f"Epoch {epoch}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f}  Val Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)
            print(f"  --> Сохранена лучшая модель (Dice: {best_dice:.4f})")
        
if __name__=="__main__":
    train()