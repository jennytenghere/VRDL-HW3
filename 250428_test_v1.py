import csv
import cv2 as cv
import gc
import json
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
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

        if type == "test":
            json_file_name = \
                self.image_folder_dir + "/test_image_name_to_ids.json"

            with open(json_file_name, "r", encoding="utf-8") as file:
                self.data = json.load(file)
            self.img_name_list = [item["file_name"] for item in self.data]
            self.ids = [item["id"] for item in self.data]
            self.folder_path = os.path.join(
                self.image_folder_dir, "test_release")
        self.img_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        img_folder_path = os.path.join(
            self.folder_path, self.img_name_list[idx])

        if self.type == "train" or self.type == "validation":
            img_path = os.path.join(img_folder_path, "image.tif")
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.open(img_folder_path).convert('RGB')
        image = self.img_transform(image)

        if self.type == "train" or self.type == "validation":
            masks = []
            labels = []
            boxes = []

            mask_name_list = [
                "class1.tif", "class2.tif", "class3.tif", "class4.tif"]
            # class index starts from 1
            for class_idx, mask_name in enumerate(mask_name_list, start=1):
                mask_path = os.path.join(img_folder_path, mask_name)
                if os.path.exists(mask_path):
                    mask = sio.imread(mask_path)
                    mask = np.array(mask, dtype=np.uint8)

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
        return image, self.ids[idx]


def a_fn(batch):
    return tuple(zip(*batch))


def test(model, test_loader, json_path):
    results = []
    model.eval()

    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc="Evaluating"):
            images = [image.to(device) for image in images]
            outputs = model(images)
            for i in range(len(outputs)):
                image_id = image_ids[i]
                box = outputs[i]['boxes'].cpu().numpy()
                score = outputs[i]['scores'].cpu().numpy()
                label = outputs[i]['labels'].cpu().numpy()
                masks = outputs[i]['masks'].cpu().numpy()
                for j in range(len(box)):
                    if score[j] > 0.05:
                        x_min, y_min, x_max, y_max = box[j]
                        binary_mask = masks[j, 0] > 0.3
                        arr = np.asfortranarray(binary_mask).astype(np.uint8)
                        rle = mask_utils.encode(arr)
                        rle['counts'] = rle['counts'].decode('utf-8')

                        one_result = {
                            "image_id": int(image_id),
                            "bbox": [float(x_min), float(y_min),
                                     float(x_max-x_min), float(y_max-y_min)],
                            "score": float(score[j]),
                            "category_id": int(label[j]),
                            "segmentation": rle
                        }
                        results.append(one_result)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    # load the dataset
    test_dataset = CustomImageDataset("test")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4,
        shuffle=False, num_workers=4,
        collate_fn=a_fn)

    # load the model
    model = models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)
    num_classes = 5
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor \
        = models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes)
    model.roi_heads.mask_predictor = \
        models.detection.mask_rcnn.MaskRCNNPredictor(
            model.roi_heads.mask_predictor.conv5_mask.in_channels,
            model.roi_heads.mask_predictor.conv5_mask.out_channels,
            num_classes
        )

    print(model)
    state_dict = torch.load("./250422_v1_all_best59.pth")
    model.load_state_dict(state_dict)
    # send to gpu
    model.to(device)

    test(model, test_dataloader, "./test-results.json")
