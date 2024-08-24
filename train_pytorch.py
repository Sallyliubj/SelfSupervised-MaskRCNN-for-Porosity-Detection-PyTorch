import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as T
from torchvision.transforms import functional as F, InterpolationMode, transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from pycocotools.coco import COCO
import numpy
import skimage
from PIL import Image
import matplotlib.pyplot as plt
import cv2
# torch.set_num_threads(4)
# torch.set_num_interop_threads(4)
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ['PYTORCH_USE_CUDA_DSA'] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING']="1"


# Define dataset class
class ForamPoreDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        # create a coco object of annotation
        self.coco = COCO(annotation)
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.ids = list(sorted(self.coco.imgs.keys()))


    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        # load images and masks
        # Use os.path.join to construct the full path to the image
        img_path = os.path.join(self.root, path)
        img = skimage.io.imread(img_path)

        if img.ndim == 2:  # grayscale image
            img = skimage.color.gray2rgb(img)
        elif img.shape[-1] == 4:  # RGBA image
            img = img[..., :3]  # remove alpha channel

        img = img.astype(numpy.float32)  # Convert to float32 for processing
        img = (img-img.min())/(img.max()-img.min())  # Rescale to [0, 1]
        
        # Convert NumPy array to PIL Image
        img = Image.fromarray((img * 255).astype(numpy.uint8))

        num_objs = len(anns)
        boxes = []
        masks = []
        labels = []

        for i in range(num_objs):
            xmin, ymin, width, height = anns[i]['bbox']
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
            masks.append(coco.annToMask(anns[i]))
            labels.append(anns[i]['category_id'])

        if num_objs > 0:
            masks = numpy.stack(masks, axis=0)
        else:
            masks = numpy.zeros((0, img.shape[0], img.shape[1]), dtype=numpy.uint8)

        # convert everything into a numpy.ndarray
        boxes = numpy.array(boxes)
        labels = numpy.array(labels)
        masks = numpy.array(masks)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id], dtype=torch.int64)
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img,  target)
        
        return img, target

    def __len__(self):
        return len(self.ids)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if numpy.random.rand() < self.flip_prob:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    keypoints = _flip_coco_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints
        return image, target


def get_transform(train):
    transforms = []
    transforms.append(lambda img, target: (T.ToTensor()(img), target))  # Handle ToTensor separately
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    
    return Compose(transforms)
        
def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


# Load a pre-trained Mask R-CNN model
def load_model(num_classes):
    # Load the pre-trained model
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Replace the head with a new one for our dataset (if necessary)
    num_classes = 2  # Including the background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    # Set the number of detections per image
    model.roi_heads.detections_per_img = 500

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def save_model(model, epoch, path):
    torch.save(model.state_dict(), os.path.join(path, f"model_epoch_{epoch}.pth"))

# Training and evaluation
def train_model(model, num_epochs, optimizer,lr_scheduler, dataset, device, save_path, loss_hist_file_path):
    
    #print(device, flush=True)
    ALL_TRAIN_LOSS = []
    ALL_VAL_LOSS = []
    
    for epoch in range(num_epochs):
        # Shuffle dataset indices at the start of each epoch
        indices = numpy.random.permutation(len(dataset))
        split = int(0.2 * len(indices))
        val_indices, train_indices = indices[:split], indices[split:]

        train_subsampler = SubsetRandomSampler(train_indices)
        val_subsampler = SubsetRandomSampler(val_indices)

        data_loader_train = DataLoader(dataset, batch_size=2, sampler=train_subsampler, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
        data_loader_val = DataLoader(dataset, batch_size=2, sampler=val_subsampler, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

        # Training phase
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for i, (images, targets) in enumerate(data_loader_train):
            images = list(image.to(device).float() for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            print(f"Batch {i + 1}:")
                
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()

            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += losses.item()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            current_lr = lr_scheduler.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}, Step {i + 1}, Train Loss: {losses.item()}, Learning Rate: {current_lr}")
            
            # Update the learning rate
            #lr_scheduler.step(losses.item())
    
        print(f"End of Training for Epoch {epoch+1}")
        lr_scheduler.step()
      
        # Save the model weights
        os.makedirs(save_path, exist_ok=True)
        #torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch+1}.pt"))

        torch.save({'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss / len(data_loader_train),
            'lr': lr_scheduler.state_dict()
             }, os.path.join(save_path, f"model_epoch_{epoch+1}.pt"))
        
        print(f"Start Validation for Epoch {epoch+1}")
        # Validation phase
        val_loss = 0
        with torch.no_grad():
            for images, targets in data_loader_val:
                images = list(image.to(device).float() for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
                #model.train()
                # Get the loss components
                loss_dict = model(images, targets)
                # Sum the loss components to get the total loss
                losses = sum(loss for loss in loss_dict.values())
                
                val_loss += losses.item()
            
        # Record train and val loss after one epoch
        ALL_TRAIN_LOSS.append(train_loss / len(data_loader_train))
        ALL_VAL_LOSS.append(val_loss / len(data_loader_val))
        
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(data_loader_train)}, Val Loss: {val_loss / len(data_loader_val)}")


    # After all training epochs finished, save the loss
    # print(f"ALL_TRAIN_LOSS: {ALL_TRAIN_LOSS}")
    # print(f"ALL_VAL_LOSS: {ALL_VAL_LOSS}")
    

    os.makedirs(loss_hist_file_path, exist_ok=True)
    
    # Open the file in write mode
    with open(loss_hist_file_path, 'w') as file:
        # Write a header and the first list
        file.write("TRAIN_LOSS:\n")
        # Write the first list to the file
        file.write(' '.join(map(str, ALL_TRAIN_LOSS)) + '\n')

        # Write a separator or empty line
        file.write('\n')  # Optional empty line for spacing

        
        file.write("VAL_LOSS:\n")
        # Write the second list to the file
        file.write(' '.join(map(str, ALL_VAL_LOSS)) + '\n')
    
    print(f"Data has been written to {loss_hist_file_path}")





# Main function
def main():

    print("CUDA available: ", torch.cuda.is_available())
    print("Number of GPUs: ", torch.cuda.device_count())
    print("CUDA version: ", torch.version.cuda)
    print("CUDNN version: ", torch.backends.cudnn.version())
    
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #print(device, flush=True)

    # our dataset has two classes only - background and foram pore
    num_classes = 2
    learning_rate = 1e-4
    num_epochs = 75

    # Root directory of the project
    ROOT_DIR = os.getcwd()
    
    # Directory of images to run detection on
    DATA_DIR = os.path.join(ROOT_DIR, "train")

    save_path = os.path.join(ROOT_DIR,'logs')
    # Create save directory if it does not exist
    os.makedirs(save_path, exist_ok=True)

    # Set the root directory for images and paths to annotation files
    train_root = DATA_DIR
    train_annotation = os.path.join(DATA_DIR, 'train.json')
    dataset = ForamPoreDataset(train_root, train_annotation, get_transform(train=True))
    
    # get the model using our helper function
    model = load_model(num_classes)

    # # move model to the right device
    # print(f"{device} before model", flush=True)
    model.to(device)
    # print(f"{device} after model", flush=True)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    #optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0001)

    # and a learning rate scheduler
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Store the loss history after training
    loss_hist_file_path = os.path.join(ROOT_DIR,'loss_hist')
    
   # let's train the model
    #print(f"{device} before train", flush=True)
    train_model(model, num_epochs, optimizer,lr_scheduler, dataset, device, save_path, loss_hist_file_path)



if __name__ == "__main__":
    main()
