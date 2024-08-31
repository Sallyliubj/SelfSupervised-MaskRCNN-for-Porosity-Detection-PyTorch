import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from pycocotools.coco import COCO
import numpy
import skimage
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import train_pytorch
import inspect_model
from train_pytorch import load_model, train_model
from coco_json import process_masks
from train_pytorch import ForamPoreDataset, Compose, RandomHorizontalFlip
from train_pytorch import get_transform
from update_dir import store_predictions, combine_masks_with_or_operation, update_inpaint_image, save_inpaint_images, create_new_folder_and_copy_images, copy_original_images, overlay_masks, save_overlay_masks

# Root directory of the project
ROOT_DIR = os.getcwd()
# Directory of images to run detection on
DATA_DIR = os.path.join(ROOT_DIR, "Reinforcement")
os.makedirs(DATA_DIR, exist_ok=True)


#<----------- The Complete Process of Training and Reinforcement ----------->#

# Load the pretrained model on COCO dataset
num_classes = 2
model = load_model(num_classes)
print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

num_classes = 2
learning_rate = 1e-4
num_epochs = 10

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

# Assume no user reinforcement from the beginning
user_refinement = False
# Maximum loop to stop the entire process
max_loop = 10

# Move train data into loop_1/in
src_folder = os.path.join(ROOT_DIR, "train")
dest_folder = os.path.join(DATA_DIR, "loop_1/in")
create_new_folder_and_copy_images(src_folder, dest_folder)

# Move Original images to loop_1's in_folder
src_folder = os.path.join(ROOT_DIR, "train")
dest_folder = os.path.join(DATA_DIR, "loop_1/in")
copy_original_images(src_folder, dest_folder)

# Update COCO-JSON file
train_folder_path = os.path.join(DATA_DIR, "loop_1/in")
train_json_path = os.path.join(DATA_DIR, "loop_1/in/via_region_data.json")
process_masks(train_folder_path, train_json_path)



for loop in range (1, max_loop):
    # Train the model
    train_root = os.path.join(ROOT_DIR, f"Reinforcement/loop_{loop}/in")
    train_annotation = os.path.join(train_root, "via_region_data.json")
    dataset = ForamPoreDataset(train_root, train_annotation, get_transform(train=True))
    
    save_path = os.path.join(ROOT_DIR,f"Reinforcement/logs/loop_{loop}")
    loss_hist_file_path = os.path.join(ROOT_DIR,f"Reinforcement/loss_hist/loop_{loop}.txt")
    
    model.to(device)
    num_epochs = 30 # suppose we train the model for a fixed number of epoches duringe each loop
    train_model(model, num_epochs, optimizer, lr_scheduler, dataset, device, save_path, loss_hist_file_path)
    
    # Store prediction results for all train data
    dataset = ForamPoreDataset(train_root, train_annotation, get_transform(train=False))
    store_predictions(model, dataset, loop, device, ROOT_DIR)
    
    # Combine all mask-in's with prediction masks above 95 confidence score and save them as mask out
    score_threshold=0.95
    combine_masks_with_or_operation(loop, score_threshold, 'cuda', ROOT_DIR)
    
    if not user_refinement:
        # Update and save inpaint images to the out-folder of the current loop
        in_folder = os.path.join(DATA_DIR, f"loop_{loop}/in")
        out_folder = os.path.join(DATA_DIR, f"loop_{loop}/out")
        save_inpaint_images(in_folder, out_folder)
    else:
        # TODO: implement user refinement interface on randomly selected images
        # TODO: update mask-out for the selected images
        # TODO: update inpaint images for the selected images
        pass
    
    # Move Updated Masks and Inpainting images from the current loop's out folder to the next loop's in folder
    src_folder = os.path.join(DATA_DIR, f"loop_{loop}/out")
    dest_folder = os.path.join(DATA_DIR, f"loop_{loop+1}/in")
    create_new_folder_and_copy_images(src_folder, dest_folder)

    # Move Original images to the next loop's in folder
    src_folder = os.path.join(DATA_DIR, f"loop_{loop}/in")
    dest_folder = os.path.join(DATA_DIR, f"loop_{loop+1}/in")
    copy_original_images(src_folder, dest_folder)

    # Update COCO-JSON file
    train_folder_path = os.path.join(DATA_DIR, f"loop_{loop+1}/in")
    train_json_path = os.path.join(DATA_DIR, f"loop_{loop+1}/in/via_region_data.json")
    process_masks(train_folder_path, train_json_path)

    # Overlay mask-in's and mask-out's in the loop and save to the "overlay_masks" folder
    save_overlay_masks(ROOT_DIR, loop)