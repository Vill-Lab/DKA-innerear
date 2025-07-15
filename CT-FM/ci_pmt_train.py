# Imports
import torch
from lighter_zoo import SegResEncoder
from models.frameworks.pmtsegresnet import PromptSegResEncoder
from monai.transforms import (
    Compose, LoadImage, EnsureType, Orientation,
    ScaleIntensityRange, CropForeground
)
from monai.inferers import SlidingWindowInferer
import argparse
# print(torch.cuda.is_available())
# print(torch.cuda.current_device())
# print(torch.cuda.device_count())
# exit()
# Parse command-line arguments

parser = argparse.ArgumentParser(description="Train a model with specified fold.")
parser.add_argument("--fold", type=int, default=0, help="Fold index for cross-validation (default: 0)")
parser.add_argument("--pmtlength", type=int, default=48, help="Prompt length (default: 48)")
parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
args = parser.parse_args()

model = PromptSegResEncoder.from_pretrained(
    "./ct_fm_feature_extractor",
    prompt_length=args.pmtlength,
)

# Preprocessing pipeline
preprocess = Compose([
    LoadImage(ensure_channel_first=True),  # Load image and ensure channel dimension
    EnsureType(),                         # Ensure correct data type
    Orientation(axcodes="SPL"),           # Standardize orientation
    # Scale intensity to [0,1] range, clipping outliers
    ScaleIntensityRange(
        a_min=-1024,    # Min HU value
        a_max=2048,     # Max HU value
        b_min=0,        # Target min
        b_max=1,        # Target max
        clip=True       # Clip values outside range
    ),
    CropForeground()    # Remove background to reduce computation
])

from torch.utils import data
import numpy as np
import os
import pandas as pd
import cv2
from torchvision.transforms import transforms
import torch
from PIL import Image
import random
import torchio as tio
import nibabel as nib
import SimpleITK as sitk

from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# # Input path
# input_path = "/home/suraj/Repositories/lighter-ct-fm/semantic-search-app/assets/scans/s0114.nii.gz"

from utils.ci2dataset import Ci2Dataset
from utils.cidataset import CiDataset
# train_dataset = Ci2Dataset("/mnt/VILL/wwx/FinalData2", "/mnt/VILL/wwx/FinalData2/ci2label.csv", "/mnt/VILL/wwx/FinalData2/scaled_features.csv", "train")
# val_dataset = Ci2Dataset("/mnt/VILL/wwx/FinalData2", "/mnt/VILL/wwx/FinalData2/ci2label.csv", "/mnt/VILL/wwx/FinalData2/scaled_features.csv", "validation")
train_dataset = CiDataset("/mnt/wwxdata/FinalData1/imagesTs", "/mnt/wwxdata/FinalData1/label.xlsx", "/mnt/wwxdata/FinalData1/scaled_features.csv", "train")
val_dataset = CiDataset("/mnt/wwxdata/FinalData1/imagesTs", "/mnt/wwxdata/FinalData1/label.xlsx", "/mnt/wwxdata/FinalData1/scaled_features.csv", "validation")

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
from torch.optim.lr_scheduler import StepLR
import argparse
import logging
from datetime import datetime

# Use the fold argument
fold = args.fold
batch_size = 1
num_workers = 16
img_size = 128

# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# train_ind, val_ind = list(kf.split(train_dataset))[fold]
skf = StratifiedKFold(5, shuffle = True, random_state = 42)
train_ind, val_ind = list(skf.split(train_dataset, train_dataset.getlabels()))[args.fold]



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

# # Reduce along dim=1: any 1 becomes 1, otherwise 0
# train_labels = (train_labels == 1).any(axis=1).astype(int)
# val_labels = (val_labels == 1).any(axis=1).astype(int)

# # Count occurrences of 0 and 1
# train_label_counts = np.bincount(train_labels, minlength=2)
# val_label_counts = np.bincount(val_labels, minlength=2)

# print(f"Train dataset label distribution: ")
# print(f"0: {train_label_counts[0]}, 1: {train_label_counts[1]}")
# print(f"Validation dataset label distribution: ")
# print(f"0: {val_label_counts[0]}, 1: {val_label_counts[1]}")

# exit()

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

import torch.optim as optim
from models.heads.classification_head import ClassificationHead  # Assuming `Head` is the head model you want to use

# Specify device with a specific GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)

# Initialize head model
head = ClassificationHead(512, 128, 2).to(device)  # Assuming 512 is the number of features extracted by the model\

trainable_params = 0
frozen_params = 0

for name, param in model.named_parameters():
    if "sva" in name or "prompt" in name or "radiomics" in name:
        param.requires_grad = True
        trainable_params += param.numel()
        print(f"Trainable: {name} - {param.numel()} parameters")
    else:
        param.requires_grad = False
        frozen_params += param.numel()
        print(f"Frozen: {name} - {param.numel()} parameters")

print(f"Total trainable parameters: {trainable_params}")
print(f"Total frozen parameters: {frozen_params}")
print(f"Total parameters: {trainable_params + frozen_params}")

for name, param in head.named_parameters():
    param.requires_grad = True

# Define optimizer and loss function
optimizer = optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-4, weight_decay=args.wd)  # Using Adam optimizer with weight decay
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)  # StepLR scheduler
criterion = CrossEntropyLoss().to(device)

# Create result directory with fold and learning rate in the name
lr = 1e-4  # Learning rate
result_dir = f"../SamResult/CT_FM_ci_prompt_pl{args.pmtlength}_wd{args.wd}_lr{lr}_fold{fold}"
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

# Training loop
epochs = 100 
for epoch in range(epochs):
    epoch_loss = 0.0
    model.train()
    head.train()
    for batch in train_dataloader:
        input_tensor, label, feats = batch
        input_tensor = input_tensor.to(device)
        input_tensor = F.interpolate(input_tensor, (img_size, img_size, img_size), mode="trilinear", align_corners=False)
        label = label.to(device)
        feats = feats.to(device, dtype=torch.float32)
        # Forward pass through the model
        output = model(input_tensor, feats)[-1]
        # Forward pass through the head
        optimizer.zero_grad()
        predictions = head(output)

        # Compute loss
        loss = criterion(predictions, label)
        epoch_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Step the scheduler
    scheduler.step()

    avg_loss = epoch_loss / len(train_dataloader)
    logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Validation loop   
    model.eval()
    head.eval()
    all_labels = []
    all_predictions = []
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_dataloader:
            input_tensor, label, feats = batch
            input_tensor = input_tensor.to(device)
            input_tensor = F.interpolate(input_tensor, (img_size, img_size, img_size), mode="trilinear", align_corners=False)
            label = label.to(device)
            feats = feats.to(device, dtype=torch.float32)
            # Forward pass through the model
            output = model(input_tensor, feats)[-1]
            # Forward pass through the head
            predictions = head(output)

            # Compute loss
            loss = criterion(predictions, label)
            val_loss += loss.item()

            # Collect predictions and labels
            all_predictions.append(predictions.cpu())
            all_labels.append(label.cpu())

    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    # Convert predictions to probabilities and binary labels
    probabilities = torch.softmax(all_predictions, dim=1)
    predicted_labels = torch.argmax(all_predictions, dim=1)
    # Save probabilities and predicted labels in .npy format
    probs_save_path = os.path.join(result_dir, f"valprobs{epoch}.npy")
    labels_save_path = os.path.join(result_dir, f"vallabels{epoch}.npy")
    np.save(probs_save_path, probabilities.cpu().numpy())
    np.save(labels_save_path, all_labels.cpu().numpy())

    # Compute metrics
    acc = accuracy_score(all_labels, predicted_labels)
    auc = roc_auc_score(all_labels, probabilities[:,1,...])
    f1 = f1_score(all_labels, predicted_labels)
    recall = recall_score(all_labels, predicted_labels)
    precision = precision_score(all_labels, predicted_labels)
    tn, fp, fn, tp = confusion_matrix(all_labels, predicted_labels).ravel()
    specificity = tn / (tn + fp)

    avg_val_loss = val_loss / len(val_dataloader)
    metrics_log = (
        f"Validation Metrics - Loss: {avg_val_loss:.4f}, Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, "
        f"Recall: {recall:.4f}, Precision: {precision:.4f}, Specificity: {specificity:.4f}"
    )
    logging.info(metrics_log)
    print(metrics_log)

logging.info("Training completed")

print("✅ Training completed")

print("✅ Feature extraction completed")
