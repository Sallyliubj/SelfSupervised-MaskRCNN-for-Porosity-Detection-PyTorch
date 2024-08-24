import numpy as np
import pandas as pd
import scipy as sp
import skimage
import skimage.io as skio
import skimage.morphology as skmo
import skimage.measure as skm
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, Button, fixed, BoundedIntText, FloatRangeSlider
import cv2
import random
import os

def combine(img1,img2,mask):
    """Combine the two images based on the mask

    Args:
        img1 (array): 
        img2 (array): origial image we want to put mask on
        mask (array): the mask

    Returns:
        np.array: the combined image
    """
    img1 = img1 * mask 
    img2 = img2 * (1-mask)
    return img1 + img2


def list_images(image_dir):
    # List all image files in the directory (adjust extensions as needed)
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.tif', '.png', '.jpg'))]
    image_files = sorted(image_files)

    # Initialize an empty list to store the images
    images = []

    # Read each image and append to the list
    for file_name in image_files:
        image_path = os.path.join(image_dir, file_name)
        image = skio.imread(image_path)
        images.append(image)
    
    return images


def remove_outer_noise(img):
    """Reduce noise from the background region outside the shell.

    Args:
        img (np.array): the target image
        threshold (float, optional): The threshold to create a binary mask on the shell. Defaults to 0.4.
        
    Returns:
        output_img (np.array)
    """
    # Apply gaussian filter to reduce noise from the region outside the shell
    gaussian_img = skimage.filters.gaussian(img, 20)
    # Convert image to uint8
    gaussian_img = (gaussian_img*255).astype(np.uint8)

    # Apply otsu thresholding
    _, otsu_thresh = cv2.threshold(gaussian_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Create mask for the outer region
    outer_region = gaussian_img < otsu_thresh
    # Dilate to further remove dark spots in background
    outer_region = skmo.dilation(outer_region,skmo.disk(6))
    # Remove noise from the region outside the shell
    output_img = img * outer_region
    
    return output_img


def get_region_within_edge(img, canny_sigma = 2, area_threshold = 300):
    """Use Canny Edge Detection and area_closing to find out the pore region

    Args:
        img (np.array): the target image
        canny_sigma (int):  sigma value for canny edge detection. Defaults to 2.
        area_threshold: area_threshold for area_closing. Defaults to 300.
    Returns:
        segmented_holes
    """
    # Apply canny edge detection to detect edges
    edges = skimage.feature.canny(img, canny_sigma)
    # Apply closing to remove dark spots and make the edge continuous
    segmented_holes = skmo.closing(edges, skmo.disk(3))
    # Apply area_closing to close any connected hole of area 300 or less. This will result in edges not only containing the outer cirumference of the hole, but the segmentation mask of the hole itself
    segmented_holes = skmo.area_closing(segmented_holes, area_threshold)

    return segmented_holes

def get_info(img_regionprops):
    
    # List of properties to inspect
    properties = ['area', 'eccentricity', 'extent']

    shell_info = []

    for prop in img_regionprops:
        region_info = {}
        for prop_name in properties:
            region_info[prop_name] = getattr(prop, prop_name)
        shell_info.append(region_info)
        
    return shell_info    


def calculate_boxplot_stats(df):
    """ 
    """
    # Calculate quartiles
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    median = df.median()
    
    # Calculate outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df < lower_bound) | (df > upper_bound)]
    
    return {
        'median': median,
        'IQR': IQR,
        'Q1': Q1,
        'Q3': Q3,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'top_5_outliers': outliers[:5]
    }
    
    
def find_threshold(shell_info):
    """ Find the threshold for area, eccentricity, and extent
    """
    df = pd.DataFrame(shell_info)
    stat = {}
    
    # area_threshold = [lower_bound,upper_bound]
    area_info = calculate_boxplot_stats(df['area'])
    stat['area_max'] = area_info['upper_bound']
    stat['area_min'] = area_info['lower_bound']
    # eccentricity_threshold = [Q1,Q3]
    eccentricity_info = calculate_boxplot_stats(df['eccentricity'])
    stat['eccentricity_max'] = eccentricity_info['Q3']
    stat['eccentricity_min'] = eccentricity_info['Q1']
    # extent_threshold = [Q1, Q3]
    extent_info = calculate_boxplot_stats(df['extent'])
    stat['extent_max'] = extent_info['Q3']
    stat['extent_min'] = extent_info['Q1']
    
    return stat

def calculate_pore_distances_and_plot(props):
   # Extract centroids and min/max axis lengths
    centroids = np.array([pore.centroid for pore in props])
    min_axes = np.array([pore.minor_axis_length for pore in props])
    max_axes = np.array([pore.major_axis_length for pore in props])
    # Compute pairwise distances between centroids
    distances = sp.spatial.distance.cdist(centroids, centroids)
    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(distances, np.inf)
    # Find the closest distance for each pore
    closest_distances = np.min(distances, axis=1)
    # Calculate the adjustment factor for each pore
    adjustment_factors = (min_axes + max_axes) / 4
    # Calculate the final result
    result = closest_distances - adjustment_factors
    # Calculate Q1, Q3, and IQR
    q1 = np.percentile(result, 25)
    q3 = np.percentile(result, 75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
 
    return int(q1) // 2



def process_image(ROOT_DIR, images, idx, new_idx):
    """
    Prepare a dataset containing original images, goood pores masks, classic masks, and inpainted images for the mask r-cnn model to train on.
    
    Parameters:
        ROOT_DIR (string): root directory
        images (list): list of images as numpy array
        idx (int): original indices from the images list
        new_idx (int): New index assigned to the train images
    
    Returns: 
        None
    """
    #The counter is a way to track the file naming. The format for the dataset elements is currently as follows: img{counter}.tif and its mask is mask{counter}.tif Counter is incremented when the save button is pushed. It's global such that it can be manually run in
    #a seperate cell to control what index we want our data to be generated from
    global counter, idx_slider
    
    # Extracts the image from the array of images generated in one of the cells above
    img = images[idx]
    # Crop the annotations in the button if needed
    #img = img[:int(img.shape[0]*0.92),:]
    
    # Remove noise from outer region of the shell
    shell_img = remove_outer_noise(img)
    
    # Find hole region inside the shell based on canny edge detection
    classic_mask = get_region_within_edge(shell_img, canny_sigma = 2, area_threshold = 300)
    
    # Propose regions inside the connected edges 
    img_label = skimage.measure.label(classic_mask)
    img_regionprops = skimage.measure.regionprops(img_label)
    
    # Find threshold values for area, eccentricity, and extent based on regionprops
    shell_info = get_info(img_regionprops)
    threshold = find_threshold(shell_info)

    # Filter through the regionprops. Find the good pores based on area, eccentricity, and extent
    good_pores = []
    good_pores_count = 0
    
    min_eccentricity = threshold['eccentricity_min']
    max_eccentricity = threshold['eccentricity_max']
    min_area = threshold['area_min']
    max_area = threshold['area_max']
    min_extent = threshold['extent_min']
    max_extent = threshold['extent_max']
    
    print(f'Area Range [{min_area},{max_area} ]')
    print(f'Eccentricity Range [{min_eccentricity},{max_eccentricity}]')
    print(f'Extent Range [{min_extent},{max_extent}]')
    
    # Create a binary mask for good pores
    pores_mask = np.zeros_like(shell_img)
    
    for prop in img_regionprops:
        #p,q,r are the conditions we expect pores to have. change them if needed
        p = min_eccentricity <= prop.eccentricity <= max_eccentricity
        q = min_extent <= prop.extent <= max_extent
        r = min_area <= prop.area <= max_area
        # Get the coordinates of the region
        coords = prop.coords
        if p and q and r:
            good_pores.append(prop)
            good_pores_count += 1
            # Fill the region in the mask
            for coord in coords:
                pores_mask[coord[0], coord[1]] = classic_mask[coord[0], coord[1]]
        
    pores_mask = skimage.filters.median(pores_mask,skmo.disk(2))
    print(good_pores_count)
    
    dilation_value1 = calculate_pore_distances_and_plot(img_regionprops)
    inpaint_mask = skmo.dilation(classic_mask, skmo.disk(dilation_value1)) # we also want to include the region slightly outside the holes
    inpaint_mask = skmo.area_closing(inpaint_mask, 200) # make sure all inpainting regions are connected
    inpainted = skimage.restoration.inpaint_biharmonic(shell_img*1.0, inpaint_mask)
    # let the good holes not inpainted
    not_inpainted = skmo.dilation(pores_mask, skmo.disk(4))
    # now the image inpainted all the hole except the good holes
    inpainted = combine(shell_img, inpainted, not_inpainted)
    
    # Display the results for user interaction
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    axes[0,0].imshow(shell_img, cmap="gray")
    axes[0,0].set_title("original shell")
    axes[0,1].imshow(inpainted, cmap="gray")
    axes[0,1].set_title("inpainted shell")
    axes[1,0].imshow(pores_mask, cmap="gray")
    axes[1,0].set_title("Pores mask")
    axes[1,1].imshow(np.clip(inpainted + (255*pores_mask),0,255), cmap = "gray")
    axes[1,1].set_title("pores detected")
    
    plt.tight_layout()
    
    save_images(ROOT_DIR, shell_img, inpainted, pores_mask, classic_mask, new_idx)
    
    
    def save_images(ROOT_DIR, shell_img, inpainted, pores_mask, classic_mask, new_idx):
        """
        Save the original image, inpainted image, good pores mask, and classic mask and assign a new index by sequence
        """
        img_filename = os.path.join(ROOT_DIR, f'training/img{new_idx}.tif')
        mask_filename = os.path.join(ROOT_DIR,f'training/mask{new_idx}.tif')
        original_filename = os.path.join(ROOT_DIR,f'training/original{new_idx}.tif')
        classic_filename = os.path.join(ROOT_DIR,f'training/classic{new_idx}.tif')
        if not os.path.exists("training"):
            os.makedirs("training")
        skio.imsave(img_filename, inpainted)
        skio.imsave(mask_filename, pores_mask)
        skio.imsave(original_filename, shell_img)
        skio.imsave(classic_filename, classic_mask)
        print(f"Images saved successfully as {img_filename} and {mask_filename}.")
    

def main():
    ROOT_DIR = os.getcwd()
    # Directory to the original image dataset
    img_dir = os.path.join(ROOT_DIR, "images")

    images = list_images(img_dir)

    # Randomly select a specific number of images from the list of images as train data
    count = 100 # adjust the value as needed
    selected_indices = random.sample(range(len(images)), count)

    new_idx = 1 # index assigned to train data (from 1 to 100)
    for idx in selected_indices:
        process_image(ROOT_DIR, images, idx, new_idx)
        new_idx += 1


if __name__ == "__main__":
    main()