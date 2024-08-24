import os
import sys
import re
import json
import shutil
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
import numpy as np
import skimage
from PIL import Image
import skimage.io as skio
from PIL import Image
import matplotlib.pyplot as plt
import cv2


from coco_json import process_masks
from train_pytorch import ForamPoreDataset, Compose, RandomHorizontalFlip
from train_pytorch import get_transform

# Root directory of the project
ROOT_DIR = os.getcwd()
# Directory of images to run detection on
DATA_DIR = os.path.join(ROOT_DIR, "Reinforcement")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "Reinforcement/logs")


def store_predictions(model, dataset, loop, device, ROOT_DIR):
    """
    Store prediction results as JSON files into the "pred" folder in each loop
    """
    for image, target in dataset:
        image_id = target['image_id'].item()
        image = ((image-image.min())/(image.max()-image.min())).float() 
    
        # Add a batch dim for prediction
        image_tensor = image.unsqueeze(0).to(device)  # Add batch dimension
        print(f"Image tensor shape: {image_tensor.size()}, {image_tensor.dtype}")
        
        # Prediction on the image
        model.eval()
        with torch.no_grad():
            predictions = model(image_tensor)
        pred = predictions[0]
    
        save_path = os.path.join(ROOT_DIR, f"Reinforcement/loop_{loop}/pred")
        # Create the output directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
       
        # Extract the necessary components
        boxes = pred['boxes'].cpu().numpy()  # Convert to numpy array
        scores = pred['scores'].cpu().numpy()
        masks = pred['masks'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()


        print(f"Number of holes detected: {len(masks)}")
         # Save the results as a dictionary
        prediction_result = {
            'boxes': boxes.tolist(),
            'scores': scores.tolist(),
            'masks': masks.tolist(),
            'labels': labels.tolist(),
        }
    
        # Construct the filename
        filename = os.path.join(save_path, f'pred_{image_id}.json')
        
        # Save the prediction result to a JSON file
        with open(filename, 'w') as f:
            json.dump(prediction_result, f)
    
        print(f'Saved Prediction Results for {filename}')
    
    print("Saving Completed for Prediction Results")


def combine_masks_with_or_operation(loop, score_threshold, device, ROOT_DIR):
    """
    Combine all mask-in's with prediction masks above score_threshold and save them as mask-out
    """
    # Set the directory and paths for input predictions, previous masks, and output masks
    pred_path = os.path.join(ROOT_DIR, f"Reinforcement/loop_{loop}/pred")
    in_path = os.path.join(ROOT_DIR, f"Reinforcement/loop_{loop}/in")
    out_path = os.path.join(ROOT_DIR, f"Reinforcement/loop_{loop}/out")
    
    # Create the output directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)

    # Loop through all prediction files in the directory
    for pred_file in os.listdir(pred_path):
        # Process each image
        if pred_file.endswith('.json'):
            print(f"File Name:{pred_file}")
            # Extract the mask index from the filename, (e.g., 'pred_{index}.json')
            mask_index = int(pred_file.split('_')[1].split('.')[0])
            print(f"Mask Index: {mask_index}")
            
            # Load the prediction result from the JSON file
            with open(os.path.join(pred_path, pred_file), 'r') as f:
                pred_result = json.load(f)
            
            # Convert the JSON list data back to NumPy arrays
            scores = np.array(pred_result['scores'])
            masks = np.array(pred_result['masks'])
            #print(f"Prediction Masks Shape: {masks.shape}")

            # Load the previous mask in from directory if it exists
            prev_mask_path = os.path.join(in_path, f'mask{mask_index}.tif')
            if os.path.exists(prev_mask_path):
                print(f"Previous Mask Path:{prev_mask_path}")
                prev_mask = Image.open(prev_mask_path)
                prev_mask = torch.tensor(np.array(prev_mask) > 0, device=device)
            else:
                prev_mask = None

            
            # Filter masks by the score threshold
            high_conf_indices = [i for i, score in enumerate(scores) if score >= score_threshold]
            filtered_masks = masks[high_conf_indices]


            if filtered_masks.size == 0:
                print(f"No masks found with scores above {score_threshold} for mask {mask_index}. Skipping.")
                continue  # Skip if no masks meet the threshold
            
            # Initialize a combined mask for the entire image with the same shape as individual masks
            combined_mask = torch.tensor(filtered_masks[0, 0], dtype = torch.bool, device=device)
            combined_mask = torch.zeros_like(combined_mask)  # Initialize an empty mask
            print(f"Combined Mask shape:{combined_mask.shape}, {combined_mask.dtype}")
            print(f"Previous Mask shape:{prev_mask.shape}, {prev_mask.dtype}")
            
            # Combine all filtered masks using the OR operation
            for i, mask in enumerate(filtered_masks):
                mask_tensor = torch.tensor(mask[0] > 0.5, dtype=torch.bool, device=device)
                combined_mask = combined_mask | mask_tensor  # OR operation to combine masks
                #print(f"Mask {i} Shape: {mask_tensor.shape}, dtype: {mask_tensor.dtype}")


            # Check if previous_mask and combined_mask are the same size
            if prev_mask.shape != combined_mask.shape:
                print(f"Error: Previous mask and current mask {mask_index} have different sizes. Skipping combination.")
                continue  # Skip combining this mask
            
            # Combine the previous mask with the new combined mask
            combined_mask = prev_mask | combined_mask  # OR operation with previous mask

            # Save the final combined mask as a .tif file
            if combined_mask is not None:
                combined_mask_cpu = combined_mask.cpu().numpy().astype(np.uint8) * 255
                combined_mask_img = Image.fromarray(combined_mask_cpu)
                combined_mask_img.save(os.path.join(out_path, f'mask{mask_index}.tif'))
                print("Combined mask created and saved.")
            else:
                print("No masks found above the score threshold.")

    print("Mask combination and saving completed.")



def update_inpaint_image(original_image, inpaint_image, original_mask, new_mask):
    """
    Update the inpaint image by showing the portion of the original image where the new mask is True.
    
    Parameters:
    original_image (numpy.ndarray): The original image.
    inpaint_image (numpy.ndarray): The current inpaint image.
    original_mask (numpy.ndarray): The original mask used to create the inpaint image.
    new_mask (numpy.ndarray): The new mask that should be applied to update the inpaint image.
    
    Returns:
    numpy.ndarray: The updated inpaint image.
    """
    
    # Ensure that the masks are boolean
    original_mask = original_mask.astype(bool)
    new_mask = new_mask.astype(bool)

    if original_mask.shape != new_mask.shape:
        print("Inconsistent Shape")
    
    # Create a copy of the inpaint image to avoid modifying the original
    updated_inpaint_image = inpaint_image.copy()
    
    # Update the inpaint image: replace the inpainted area with the corresponding part of the original image where new_mask is True
    updated_inpaint_image[new_mask] = original_image[new_mask]
    #print(updated_inpaint_image.dtype,updated_inpaint_image.shape, updated_inpaint_image.min(),updated_inpaint_image.max())
    
    return updated_inpaint_image



def save_inpaint_images(in_folder, out_folder):
    """
    Process all images in the in-folder, update inpaint images, and save to the out-folder
    
    Parameters:
    inpaint_folder (str): Path to the folder containing previous inpaint images and original masks.
    new_mask_folder (str): Path to the folder containing new masks and the location to save updated inpaint images.
    """
    # Get all filenames from the inpaint folder (assuming they correspond to new masks)
    filenames = os.listdir(in_folder)
    inpaint_filenames = [f for f in filenames if f.startswith("img") and f.endswith(".tif")]
    original_filenames = [f for f in filenames if f.startswith("original") and f.endswith(".tif")]
    mask_filenames = [f for f in filenames if f.startswith("mask") and f.endswith(".tif")]

    def extract_index(filename):
        # Use regex to find the digits in the filename
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else -1
    
    # Sort the filenames by their numerical index
    mask_filenames.sort(key=extract_index)
    inpaint_filenames.sort(key=extract_index)
    original_filenames.sort(key=extract_index)
    
    # Process images pairwise by index
    for mask_filename, inpaint_filename, original_filename in zip(mask_filenames, inpaint_filenames, original_filenames):
        # Extract the index and ensure filenames match
        mask_index = extract_index(mask_filename)
        inpaint_index = extract_index(inpaint_filename)
        original_index = extract_index(original_filename)
        
        if mask_index == inpaint_index == original_index:
            # Construct full file paths
            mask_path = os.path.join(in_folder, mask_filename)
            inpaint_path = os.path.join(in_folder, inpaint_filename)
            original_path = os.path.join(in_folder, original_filename)
            
            new_mask_path = os.path.join(out_folder, mask_filename)  # Assuming new masks have the same name as the old masks
            new_inpaint_path = os.path.join(out_folder, inpaint_filename)  # Save the updated inpaint image with the same name
            
            # Load the images and masks
            original_image = skio.imread(original_path)
            inpaint_image = skio.imread(inpaint_path)
            # Convert inpaint_image to [0,255] unit8
            inpaint_image = (inpaint_image/inpaint_image.max())* 255
            inpaint_image = inpaint_image.astype('uint8') 
           
            prev_mask = skio.imread(mask_path)
            new_mask = skio.imread(new_mask_path)
            
            # Update the inpaint image
            new_inpaint_image = update_inpaint_image(original_image, inpaint_image, prev_mask, new_mask)
            
            # Save the updated inpaint image to the target folder
            cv2.imwrite(new_inpaint_path, new_inpaint_image)
            print(f"Saved updated inpaint image: {new_inpaint_path}")
        else:
            print(f"Filename index mismatch for {mask_filename}, {inpaint_filename}, and {original_filename}. Skipping...")



def create_new_folder_and_copy_images(src_folder, dest_folder):
    """
    Create a new folder and copy updated inpaint images and masks from previous out-folder to the next in-folder
    """
    # Create the new folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"Created new directory: {dest_folder}")

    # List all files in the source folder
    files = os.listdir(src_folder)

    # Copy each image file to the new folder
    for file_name in files:
        # Construct full file path
        src_file_path = os.path.join(src_folder, file_name)
        dest_file_path = os.path.join(dest_folder, file_name)
        
        # Check if the file is an image
        if file_name.lower().startswith(('img', 'mask')) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif')):
            shutil.copy(src_file_path, dest_file_path)
            print(f"Copied {file_name} to {dest_folder}")


def copy_original_images(src_folder, dest_folder):
    """
    Copy original images from the previous in-folder to the next in-folder
    """
    # Create the new folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"Created new directory: {dest_folder}")

    # List all files in the source folder
    files = os.listdir(src_folder)

    # Copy each image file to the new folder
    for file_name in files:
        # Construct full file path
        src_file_path = os.path.join(src_folder, file_name)
        dest_file_path = os.path.join(dest_folder, file_name)
        
        # Check if the file is an image
        if file_name.lower().startswith('original') and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif')):
            shutil.copy(src_file_path, dest_file_path)
            print(f"Copied {file_name} to {dest_folder}")



def overlay_masks(prev_mask, new_mask):
    """
    Overlay the two masks to inspect the difference
    """
    # Create a black canvas with the same height and width as the masks
    prev_mask = prev_mask.astype(bool)
    new_mask = new_mask.astype(bool)
    
    overlay_image = np.zeros((prev_mask.shape[0], prev_mask.shape[1], 3), dtype=np.uint8)
    
    overlay_image[new_mask] = [255,125,0] # orange
    overlay_image[prev_mask] = [0,0,255] # blue

    return overlay_image

def save_overlay_masks(ROOT_DIR, loop):
    """
    Combine mask-in and mask-out, and save the overlay masks to the new folder
    
    Parameters:
    image (numpy.ndarray): The image to save (as a numpy array).
    filename (str): The filename to save the image as (including the path and extension).
    """

    prev_mask_path = os.path.join(ROOT_DIR, f"Reinforcement/loop_{loop}/in")
    new_mask_path = os.path.join(ROOT_DIR, f"Reinforcement/loop_{loop}/out")
    save_path = os.path.join(ROOT_DIR, f"Reinforcement/loop_{loop}/overlay_masks_30_epoch")
    os.makedirs(save_path, exist_ok=True)
    
    for idx in range(1,90):
        prev_mask = skio.imread(os.path.join(prev_mask_path, f"mask{idx}.tif"))
        new_mask = skio.imread(os.path.join(new_mask_path, f"mask{idx}.tif"))
        if prev_mask.shape != new_mask.shape:
            print(f"Inconsistant Shape. Previous mask has shape:{prev_mask.shape}, but new mask has shape: {new_mask.shape}")
            continue
        
        # Overlay prev_mask and new_mask
        overlay_image = overlay_masks(prev_mask, new_mask)                   
        
        # Save the image to the specified file using OpenCV
        cv2.imwrite(os.path.join(save_path, f"overlay_mask{idx}.tif"), overlay_image)

