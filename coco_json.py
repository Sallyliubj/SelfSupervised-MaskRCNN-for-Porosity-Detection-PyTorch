import os
import json
import numpy as np
import cv2
from PIL import Image
from pycocotools import mask
from tqdm import tqdm
import skimage
import re

MASK_EXT = '.tif'
IMG_EXT = '.tif'

def process_masks(folder_path, output_json_path, MASK_EXT = '.tif', IMG_EXT = '.tif'):
    coco_json = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}]
    }

    annotation_id = 1
    image_id = 1
    
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out image and mask files
    image_files = [f for f in files if f.startswith("img") and f.endswith(IMG_EXT)]
    mask_files = [f for f in files if f.startswith("mask") and f.endswith(MASK_EXT)]
    
    # Sort the files to ensure they are aligned
    # image_files.sort()
    # mask_files.sort()

    # Define a function to extract the numerical index from the filename
    def extract_index(filename):
        # Use regex to find the digits in the filename
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else -1
    
    # Sort filenames by the extracted index
    image_files = sorted(image_files, key=extract_index)
    mask_files = sorted(mask_files, key=extract_index)
    
    for img_filename, mask_filename in tqdm(zip(image_files, mask_files), total=len(image_files)):
        
        # Debugging: Print the file paths
        print(f"Processing {img_filename} and {mask_filename}")
        
        # Load image
        image_path = os.path.join(folder_path, img_filename)
        
        
        try:
            image = cv2.imread(image_path)
            
            if image.ndim == 2:  # grayscale image
                    image = skimage.color.gray2rgb(image)
            elif image.shape[-1] == 4:  # RGBA image
                image = image[..., :3]  # remove alpha channel
            
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            continue   
        
        height, width = image.shape[:2] 
        
        # Load mask 
        mask_path = os.path.join(folder_path, mask_filename)
        
        try:
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_image = (mask_image > 0).astype(np.uint8)  # ensure mask is binary
        except Exception as e:
            print(f"Error reading mask {mask_path}: {e}")
            continue      


        # Add image info to COCO JSON
        coco_json["images"].append({
            "id": image_id,
            "file_name": img_filename,
            "width": width,
            "height": height
        })
        
        # Find contours (external only)
        contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [x, y, w, h]
            
            # Create segmentation mask
            segmentation = contour.flatten().tolist()
            
            if len(segmentation) <= 4:
                print("image file", img_filename, "raises error:", segmentation)
                
                new_element = [segmentation[-1]+1]
                segmentation = bbox + new_element
                
                print(len(segmentation))
            
            # Create RLE
            rle = mask.encode(np.asfortranarray(mask_image))
            area = mask.area(rle).tolist()
            
            coco_json["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "area": area,
                "segmentation": [segmentation],
                "iscrowd": 0
            })
            annotation_id += 1
        
        image_id += 1
    
    # Save COCO JSON to file
    with open(output_json_path, 'w') as outfile:
        json.dump(coco_json, outfile, indent=4)



if __name__ == "__main__":
    # Root directory of the project
    ROOT_DIR = os.getcwd()
    
    # Directory of images to run detection on
    TRAIN_DIR = os.path.join(ROOT_DIR, "train")
    TEST_DIR = os.path.join(ROOT_DIR, "test")
    
    train_folder_path = TRAIN_DIR
    train_json_path =os.path.join(TRAIN_DIR, "via_region_data.json")
    process_masks(train_folder_path, train_json_path)
    
    test_folder_path = TEST_DIR
    test_json_path = os.path.join(TEST_DIR, "via_region_data.json")
    process_masks(test_folder_path, test_json_path)
