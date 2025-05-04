import csv
import cv2 as cv
import gc
import json
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
import scipy.ndimage
import skimage.io as sio
from sklearn.metrics import confusion_matrix
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models, ops
from torch.utils.data import Dataset
# from torchvision.models import ResNet50_Weights
from torchvision.models import resnext50_32x4d
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.misc import FrozenBatchNorm2d
import torchmetrics
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.cuda.empty_cache()

gc.collect()

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
map_metric = torchmetrics.detection.MeanAveragePrecision().to(device)


class CustomImageDataset(Dataset):
    def __init__(self, type):
        super().__init__()
        self.type = type
        self.image_folder_dir = "./hw3-data-release"
        if type == "train" or type == "validation":
            self.folder_path = os.path.join(self.image_folder_dir, "train")
            self.img_name_list = os.listdir(self.folder_path)
            self.img_name_list.sort()
            # split the dataset into training set and validation set
            split_idx = int(len(self.img_name_list) * 0.7)
            if type == "train":
                self.img_name_list = self.img_name_list[:split_idx]
            else:
                self.img_name_list = self.img_name_list[split_idx:]
        # transform image to tensor without augmentation
        self.img_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        if self.type == "train" or self.type == "validation":
            return len(self.img_name_list)

    def __getitem__(self, idx):
        img_folder_path = os.path.join(
            self.folder_path, self.img_name_list[idx])
        # load the input image
        img_path = os.path.join(img_folder_path, "image.tif")
        image = Image.open(img_path).convert('RGB')
        image = self.img_transform(image)

        masks = []
        labels = []
        boxes = []

        # segmentation ground truth images
        mask_name_list = [
            "class1.tif", "class2.tif", "class3.tif", "class4.tif"]
        # class index starts from 1
        for class_idx, mask_name in enumerate(mask_name_list, start=1):
            mask_path = os.path.join(img_folder_path, mask_name)
            # not all images have 4 segmentation images
            # if this segmentation image exist
            if os.path.exists(mask_path):
                mask = sio.imread(mask_path)
                mask = np.array(mask, dtype=np.uint8)

                # split every classified cell into one bounding box
                labeled_mask, num_instances = scipy.ndimage.label(mask > 0)
                for i in range(1, num_instances + 1):
                    instance_mask = (labeled_mask == i).astype(np.uint8)
                    pos = np.where(instance_mask)
                    if pos[0].size == 0 or pos[1].size == 0:
                        continue
                    ymin, ymax = pos[0].min(), pos[0].max()
                    xmin, xmax = pos[1].min(), pos[1].max()
                    if xmax > xmin and ymax > ymin:
                        masks.append(instance_mask)
                        labels.append(class_idx)
                        boxes.append([xmin, ymin, xmax, ymax])
        masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks
        }

        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def a_fn(batch):
    return tuple(zip(*batch))


def train(model, train_loader, optimizer, epoch):
    model = model.to(device)
    model.train()
    total_loss = 0

    for images, targets in tqdm(
        train_loader, desc=f"Epoch {epoch+1} Training"
    ):

        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # loss calculation
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}")
    print(f"Traing Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, val_loader):
    model.eval()
    map_metric.reset()

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]

            # count map
            outputs = model(images)
            map_metric.update(outputs, targets)

    map_result = map_metric.compute()

    print(f"Validation mAP: {map_result['map']:.4f}")

    return map_result["map"]


if __name__ == "__main__":
    num_epochs = 30
    version = "250422_v1"
    # make a new folder for saving the results
    if not os.path.exists("./"+version):
        os.mkdir("./"+version)

    # load the dataset
    train_dataset = CustomImageDataset("train")
    val_dataset = CustomImageDataset("validation")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4,
        shuffle=True, num_workers=4,
        collate_fn=a_fn)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4,
        shuffle=False, num_workers=4,
        collate_fn=a_fn)

    # load the model
    num_classes = 5

    resnext = resnext50_32x4d(pretrained=True, norm_layer=FrozenBatchNorm2d)
    return_layers = {
        'layer1': '0',
        'layer2': '1',
        'layer3': '2',
        'layer4': '3',
    }
    resnext_backbone = IntermediateLayerGetter(
        resnext, return_layers=return_layers)
    in_channels_list = [256, 512, 1024, 2048]
    out_channels = 256
    # default anchor
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,), (1024,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 6
    )
    # fpn_v2
    backbone_with_fpn = BackboneWithFPN(
        backbone=resnext_backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelP6P7(2048, out_channels)
    )
    # create mask r-cnn
    model = MaskRCNN(
        backbone=backbone_with_fpn,
        num_classes=5,
        rpn_anchor_generator=anchor_generator
    )

    print("parameters: ", sum(p.numel() for p in model.parameters()))
    print(model)
    # send to gpu
    model.to(device)
    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # initialization for finding highest map
    best_map = 0
    train_loss_list = []
    val_map_list = []
    # 1 epoch = training+test
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer, epoch)
        val_map = evaluate(model, val_dataloader)
        torch.save(model.state_dict(),
                   "./" + version + "/" + version + "_" + str(epoch) + ".pth")
        train_loss_list.append(train_loss)
        val_map_list.append(val_map)
        with open("./" + version + "/" + version+".txt", "a") as f:
            f.write(f"Epoch {epoch+1} {train_loss:.4f} {val_map:.4f}\n")

        if val_map > best_map:
            best_map = val_map
            torch.save(
                model.state_dict(),
                "./" + version + "/" + version + "_best"+str(epoch) + ".pth")
            print("Best model saved!")

    plt.figure(figsize=(12, 6))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1),
             train_loss_list,
             label='Train Loss',
             color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Loss of Epoch 1~{num_epochs}')

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1),
             val_map_list,
             label='Validation mAP',
             color='red')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.legend()
    plt.title(f'mAP of Epoch 1~{num_epochs}')

    plt.tight_layout()

    image_path = f"./{version}/{version}_{num_epochs}.png"
    plt.savefig(image_path)
    plt.close()

    print("Training Finished!")
