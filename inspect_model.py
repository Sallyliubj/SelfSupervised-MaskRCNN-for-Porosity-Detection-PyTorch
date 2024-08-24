import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as F
import skimage.io as skio
import numpy
import numpy as np
import skimage
import matplotlib.patches as patches
import skimage.io as skio
import cv2
import os
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from PIL import Image



def load_trained_model(model_path, device):
    """ Load the trained model by state_dict.
    """
    num_classes = 2  # (including background)
    # Load the model architecture
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    # Set the number of detections per image
    model.roi_heads.detections_per_img = 500
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    def optimizer_to(optim, device):
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    # Initialize parameters
    learning_rate = 1e-4
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    # Load the saved model weights, optimizer, lr_scheduler
    print(f"Model Path: {model_path}")
    checkpoint = torch.load(model_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Move model and optimizer to the right device
    model.to(device)
    optimizer_to(optimizer, device)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
    lr_scheduler.load_state_dict(checkpoint['lr'])

    return model, optimizer, lr_scheduler


def visualize_dataset(dataset, idx, show_bbox = False):
    """
    Visulize the image from train or test dataset by index.

    Parameters:
        dataset (object of pytorch.ForamPoreDataset)
        idx (int): index of the image from dataset
        show_bbox (bool): whether show the bounding boxes for each mask
    """
    
    img, target = dataset[idx]
    img = img.permute(1, 2, 0).numpy()  # Convert tensor image to numpy array
   
    fig, ax = plt.subplots(1, figsize=(10, 10))
    # Normalize the image
    img = ((img-img.min())/(img.max()-img.min()))
        
    print(f"Display image type: {img.dtype}, Min: {img.min()}, Max: {img.max()}")
    ax.imshow(img)

    boxes = target['boxes'].numpy()
    masks = target['masks'].numpy()

    if show_bbox:
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    for mask in masks:
        mask = mask.astype(numpy.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour[:, 0, :]  # Convert contour array shape
            polygon = patches.Polygon(contour, fill=True, color='b', alpha=0.7)
            ax.add_patch(polygon)

    print(f"Number of Holes in the foram: {len(masks)}")
    
    plt.axis('off')
    plt.show()


def crop_annotation(image):
    """
    Crop the annotation from the bottom of the image

    Parameters:
    image (Numpy Array or PIL Image): the image to be cropped

    Returns:
    image(PIL Image): the cropped image
    """
    if isinstance(image, Image.Image):
        # Calculate the new height
        new_height = int(image.height * 0.92)
        # Define the crop area (left, upper, right, lower)
        left = 0
        upper = 0
        right = image.width
        lower = new_height
        crop_area = (left, upper, right, lower)
        # Crop the image
        image = image.crop(crop_area)
    elif isinstance(image, np.ndarray):
        image = image[0:int(0.92*image.shape[0]), :]
    else:
        raise("Invalid Image Type")
    
    return image

def preprocess_image(image, device):
    """
        Transform the image to tensor and add a batch dimension for prediction

        Parameters:
        image (PIL Image or Numpy Array): no need to transpose numpy array before transform
        device:

        Returns:
        image_tensor (Batch_dim, C,H,W)
    """
  
    #print(f"Image dimension before transform: {image.shape}, {image.dtype}")
   
    # Transform the image
    transform = transforms.Compose([
        transforms.ToTensor()
    ]) 
    image_tensor = transform(image)
    #print(f"Image dimension after transform: {image_tensor.shape},{image.dtype}")
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    return image_tensor


def visualize_prediction(image, predictions):
    """
    Visualize predicted masks on the image

    Parameters:
        image (Numpy array or PIL Image or torch.Tensor)
        predictions

    """

    # Extract the predictions
    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_masks = predictions[0]['masks'].cpu().numpy()
  
    # Set a score threshold to filter predictions (lower it for debugging)
    score_threshold = 0.7
    pred_indices = np.where(pred_scores >= score_threshold)[0]
    
    # Filter boxes and masks
    filtered_boxes = pred_boxes[pred_indices]
    filtered_masks = pred_masks[pred_indices]
    filtered_scores = pred_scores[pred_indices]
    
    # Check if there are any predictions after filtering
    if len(filtered_boxes) == 0:
        print("No predictions above the score threshold.")
    else:
        # Make sure the image is numpy array
        if isinstance(image, np.ndarray):
            image_np = image
        elif isinstance(image, Image.Image):
            image_np = numpy.array(image)
            print("Convert image from PIL to numpy")
        elif isinstance(image, torch.Tensor):
            print("Convert image from tensor to numpy")
            # If it's on a GPU, move it to the CPU first
            image_np = image.cpu().permute(1,2,0).numpy()
        else:
            print("Invalid Image Shape for Display")

        # Make sure the display has type float32, range [0,1]
        if not(image_np.dtype == np.float32 and image_np.min() >= 0.0 and image_np.max() <= 1.0):
            image_np = image_np / 255
            image_np = ((image_np-image_np.min())/(image_np.max()-image_np.min())).astype('float32') # Rescale to [0, 1]

            
        print(f"Display Image Shape: {image_np.shape}, Type:{image_np.dtype}, Min:{image_np.min()}, Max: {image_np.max()}")
        
        # Plot the image
        plt.figure(figsize=(10, 10))

        count = 0
        # Plot the bounding boxes and masks
        for i in range(len(filtered_boxes)):
            box = filtered_boxes[i]
            mask = filtered_masks[i, 0]
            score = filtered_scores[i]
    
            # Draw bounding box
            # plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
            #                                   fill=False, edgecolor='red', linewidth=2))
    
            # Apply mask to the image
            mask = mask > 0.5
            if score >= 0.95:
                count += 1
                color = np.array([0, 1, 0], dtype=np.float32)  # Green
                
            else:
                color = np.array([1, 0.7, 0], dtype=np.float32)  # Orange
    
            image_np[mask] = image_np[mask] * 0.3 + color * 0.7
           
        print(f"Total Number of Holes detected:{len(pred_masks)}")
        print(f"Number of Holes detected above 0.95 score thresdhold: {count}")
        print(f"Number of Holes detected above {score_threshold} and below 0.95 score threshold: {len(filtered_scores)-count}")
        
        plt.imshow(image_np)
        plt.axis('off')
        plt.show()



