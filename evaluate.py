import torch
import matplotlib.pyplot as plt

import numpy as np
import cv2
import config
from dataset import get_dataloaders
from model import build_model

def load_model():
    model=build_model()
    checkpoint=torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def predict(model, image):
    with torch.no_grad():
        preds=model(image)
        preds=torch.sigmoid(preds)
        preds=(preds>0.5).float()
    return preds

def denormalize(image_tensor):
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    image=image_tensor.permute(1,2,0).cpu().numpy()
    
    image=std*image+mean
    image=np.clip(image,0,1)
    return image

def visualize(num_samples=4):
    _, val_loader=get_dataloaders()
    model=load_model()
    images, masks=next(iter(val_loader))
    
    images=images.to(config.DEVICE)
    preds=predict(model,images)
    
    fig,axes=plt.subplots(num_samples,3,figsize=(12,4*num_samples))
    for i in range(num_samples):
        original=denormalize(images[i])
        real_mask=masks[i].squeeze().cpu().numpy()
        
        pred_mask=preds[i].squeeze().cpu().numpy()
        
        axes[i,0].imshow(original)
        axes[i,0].set_title("Original Image")
        axes[i,0].axis("off")
        
        axes[i,1].imshow(real_mask,cmap="gray")
        axes[i,1].set_title("Ground Truth Mask")
        axes[i,1].axis("off")
        
        axes[i,2].imshow(pred_mask,cmap="gray")
        axes[i,2].set_title("Predicted Mask")
        axes[i,2].axis("off")
        
    plt.tight_layout()
    plt.savefig("results.png",dpi=150)
    print("Результаты сохранены в results.png")
    
    plt.show()
    
if __name__=="__main__":
    visualize()