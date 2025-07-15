from visualizer import get_local
get_local.activate() # 激活装饰器
import os
import warnings
import torch
from torch.utils import data
import numpy as np
import os
import pandas as pd
import cv2
import torch
from PIL import Image
import random
import torchio as tio
import nibabel as nib
import SimpleITK as sitk

import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from tqdm import tqdm

from merlin import Merlin, pmtMerlin
import argparse
import logging
from datetime import datetime

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a model with specified fold.")
parser.add_argument("--fold", type=int, default=0, help="Fold index for cross-validation (default: 0)")
parser.add_argument("--pmtlength", type=int, default=48, help="Prompt length (default: 48)")
parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
args = parser.parse_args()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# # Input path
# input_path = "/home/suraj/Repositories/lighter-ct-fm/semantic-search-app/assets/scans/s0114.nii.gz"

from utils.ci2dataset import Ci2Dataset
from utils.cidataset import CiDataset
train_dataset = CiDataset("/mnt/Data_new/wwx/FinalData1/imagesTs", "/mnt/Data_new/wwx/FinalData1/label.xlsx", "/mnt/Data_new/wwx/FinalData1/scaled_features.csv", "train")
val_dataset = CiDataset("/mnt/Data_new/wwx/FinalData1/imagesTs", "/mnt/Data_new/wwx/FinalData1/label.xlsx", "/mnt/Data_new/wwx/FinalData1/scaled_features.csv", "validation")

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
from torch.optim.lr_scheduler import StepLR

# Use the fold argument
fold = args.fold
batch_size = 1
num_workers = 16
img_size = 128

kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_ind, val_ind = list(kf.split(train_dataset))[fold]
# skf = StratifiedKFold(5, shuffle = True, random_state = 42)
# train_ind, val_ind = list(skf.split(train_dataset, train_dataset.getlabels()))[args.fold]

train_labels = [train_dataset.labels[i] for i in train_ind]
val_labels = [val_dataset.labels[i] for i in val_ind]
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
train_label_counts = np.sum(train_labels, axis=0)
val_label_counts = np.sum(val_labels, axis=0)
print(f"Train dataset label distribution: ")
print(train_label_counts)
print(f"Validation dataset label distribution: ")
print(val_label_counts)

train_dataset = Subset(train_dataset, train_ind)
val_dataset = Subset(val_dataset, val_ind)
train_sampler = None
val_sampler = None
shuffle = True

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=train_sampler,
    batch_size=batch_size, 
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    sampler=val_sampler,
    batch_size=batch_size, 
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=True,
)

## Get the Image Embeddings

import torch.nn as nn
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

head = nn.Linear(2048, 2).to(device)

# model = Merlin(ImageEmbedding=True)
model = pmtMerlin(ImageEmbedding=True, prompt_length=args.pmtlength)
model = model.to(device)

trainable_params = 0
frozen_params = 0

for name, param in model.named_parameters():
    if "sva" in name or "prompt" in name or "radiomics" in name:
        param.requires_grad = True
        trainable_params += param.numel()
        print(f"Trainable: {name} - {param.numel()} parameters")
    elif "text" in name:
        continue
    else:
        param.requires_grad = False
        frozen_params += param.numel()
        print(f"Frozen: {name} - {param.numel()} parameters")

print(f"Total trainable parameters: {trainable_params}")
print(f"Total frozen parameters: {frozen_params}")
print(f"Total parameters: {trainable_params + frozen_params}")
# exit()
for name, param in head.named_parameters():
    param.requires_grad = True

import torch.optim as optim
lr = 1e-4  # Learning rate
optimizer = optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=args.wd)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # StepLR scheduler
criterion = CrossEntropyLoss().to(device)
# alphas = [0.8, 0.94, 0.9, 0.95, 0.93, 0.92]
# from monai.losses import focal_loss
# criterion = [focal_loss.FocalLoss(to_onehot_y=True, alpha=alpha, gamma=1.0, use_softmax=True, reduction='mean') for alpha in alphas]

model_pths = [
    "/mnt/Data_new/wwx/SamResult/Merlin_ci_pmtl4_repeat_wd0.0001_lr0.0001_fold0/best_model_epoch8.pth",
    "/mnt/Data_new/wwx/SamResult/Merlin_ci2_wd0.01_lr0.0001_fold1/best_model_epoch31.pth",
    "/mnt/Data_new/wwx/SamResult/Merlin_ci2_wd0.01_lr0.0001_fold2/best_model_epoch45.pth",
    "/mnt/Data_new/wwx/SamResult/Merlin_ci2_wd0.01_lr0.0001_fold3/best_model_epoch51.pth",
    "/mnt/Data_new/wwx/SamResult/Merlin_ci2_wd0.01_lr0.0001_fold4/best_model_epoch25.pth",
]

model_weights_path = model_pths[fold]
loaded_state_dict = torch.load(model_weights_path, map_location="cpu")["model_state_dict"]
# model_state_dict = model.state_dict()
# print("loaded_state_dict keys:")
# for key in loaded_state_dict.keys():
#     print(key)
# # Print parameters that are in the model but not in the loaded state dict
# missing_keys = [key for key in model_state_dict.keys() if key not in loaded_state_dict]
# if missing_keys:
#     print("Parameters in the model but not in the loaded state dict:")
#     for key in missing_keys:
#         print(key)
model.load_state_dict(loaded_state_dict, strict=True)
# exit()
result_dir = f"../../SamResult/Merlin_ci2_visualize_wd{args.wd}_lr{lr}_fold{fold}"
os.makedirs(result_dir, exist_ok=True)
print(f"Result directory created at {result_dir}")

# Configure logging
current_time = datetime.now().strftime("%Y%m%d_%H%M")
log_file = os.path.join(result_dir, f"training_log_{current_time}.txt")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Training started")

from tqdm import tqdm
# Training loop
epochs = 1 
best_f1 = 0.0  # Initialize the best F1 score

for epoch in range(epochs):
    epoch_loss = 0.0
    # model.train()
    # head.train()
    # for batch in tqdm(train_dataloader, desc="Training Progress"):
    #     input_tensor, label, feats = batch
    #     input_tensor = input_tensor.to(device)
    #     input_tensor = F.interpolate(input_tensor, (img_size, img_size, img_size), mode="trilinear", align_corners=False)
    #     label = label.to(device)
    #     feats = feats.to(device)
    #     label = ((label == 1).any(dim=1)).long()
    #     # Forward pass through the model
    #     output = model(input_tensor, feats=feats)[0]
    #     # Forward pass through the head
    #     optimizer.zero_grad()
    #     predictions = head(output)
    #     # predictions = head(output).view(label.shape[0], 2, -1)
    #     # Compute loss
    #     loss = criterion(predictions, label)
    #     # label = label.unsqueeze(1)
    #     # print(predictions.shape, label.shape)
    #     # losses = [loss_fn(predictions[:,:,i], label[:,:,i]) for i, loss_fn in enumerate(criterion)]
    #     # loss = sum(losses)
    #     # epoch_loss += loss.item()

    #     # Backward pass and optimization
    #     loss.backward()
    #     optimizer.step()

    # # Step the scheduler
    # scheduler.step()

    # avg_loss = epoch_loss / len(train_dataloader)
    # logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    # print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Validation loop   
    model.eval()
    head.eval()
    all_labels = []
    all_predictions = []
    val_loss = 0.0

    with torch.no_grad():
        ind = 11
        # 1 6
        input_tensor, label, feats = list(val_dataloader)[ind]
        input_tensor = input_tensor.to(device)
        input_tensor = F.interpolate(input_tensor, (img_size, img_size, img_size), mode="trilinear", align_corners=False)
        label = label.to(device)
        feats = feats.to(device)
        # label = ((label == 1).any(dim=1)).long()
        # Forward pass through the model
        output = model(input_tensor, feats=feats)
        print(output)
        # Forward pass through the head
        predictions = head(output)
        # predictions = head(output).view(label.shape[0], 2, -1)

        # Compute loss
        # loss = criterion(predictions, label)
        
        cache = get_local.cache
        print(cache.keys())
        print(len(cache['SvaAttention.forward']))
        select_cache = cache['SvaAttention.forward']
        print(len(select_cache))
        img_size = 128
        target_block_ind = 0
        if target_block_ind == 0:
            H = 16
            W = 8
            D = 8
        else:
            H = 8
            W = 4
            D = 4
        head_index = 1
        target_cache = select_cache[target_block_ind]
        print(target_cache.shape)
        print(type(target_cache))
        target_cache = target_cache[:, head_index, :, :].mean(-1, keepdims=True)
        target_cache = target_cache.reshape(1, H, W, D)[0].transpose(1, 2, 0)
        # 012 120 102 201 210 021
        mask = tio.Resize((128, 128, 128))(np.expand_dims(target_cache.astype(np.float32), 0)).squeeze()
        mask = mask / np.max(mask)

        image_resized = input_tensor.cpu().numpy()[0, 0, :, :, :] * 255.0
        image_resized = image_resized.astype(np.uint8)
        from matplotlib import pyplot as plt
        for i in range(image_resized.shape[0]):
            fig, ax = plt.subplots(1, 2, figsize=(10,7))
            fig.tight_layout()
            print(i)
            image_slice = Image.fromarray(image_resized[i,:,:], mode='L')
            mask_slice = mask[i]

            # ax[0].imshow(grid_image)
            ax[0].imshow(image_slice, cmap='gray')
            ax[0].axis('off')
            
            # ax[1].imshow(grid_image)
            ax[1].imshow(image_slice, cmap='gray')
            ax[1].imshow(mask_slice, alpha=0.6, cmap='rainbow')
            ax[1].axis('off')
            plt.savefig(f'attnmap/attention_map{i}.png', bbox_inches='tight', pad_inches=0.0)
            plt.close()

