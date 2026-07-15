"""
Standalone inference module for the grocery shelf product detector.
Pulled out of product_detection.ipynb - training/eval code left behind on purpose,
this file only has what's needed to load the trained model and run it on an image.

Expects this folder layout (relative to wherever this file is run from):
    models/model_best.pth              (trained weights)
    config_files/config.json           (img_size, strides, grid_sizes, anchors - see notebook)

You can also override both paths when creating ProductDetectorRuntime(...).
"""

import json
import math

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image, ImageDraw


# ---------------- model architecture (same as notebook) ----------------

class DetectionHead(nn.Module):
    def __init__(self, in_ch, mid_ch=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 5, 1),
        )

    def forward(self, x):
        out = self.net(x)
        return out.permute(0, 2, 3, 1)  # (batch, h, w, 5)


class ProductDetector(nn.Module):
    capture_idxs = {6: 32, 13: 96, 18: 1280}

    def __init__(self, pretrained=False):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights).features
        chans = list(self.capture_idxs.values())
        self.heads = nn.ModuleList([DetectionHead(c) for c in chans])

    def forward(self, x):
        feats = []
        capture_set = set(self.capture_idxs.keys())
        out = x
        for idx, layer in enumerate(self.backbone):
            out = layer(out)
            if idx in capture_set:
                feats.append(out)
            if idx == max(capture_set):
                break
        return [head(f) for head, f in zip(self.heads, feats)]


# ---------------- decode + nms (same as notebook) ----------------

def iou_xywh(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_thresh=0.45):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        rest = idxs[1:]
        idxs = np.array([i for i in rest if iou_xywh(boxes[current], boxes[i]) < iou_thresh])
    return keep


def decode(preds, anchors, grid_sizes, img_size, conf_thresh=0.3):
    all_boxes, all_scores = [], []
    for pred, (anchor_w, anchor_h), grid in zip(preds, anchors, grid_sizes):
        cell = img_size / grid
        obj = torch.sigmoid(pred[..., 0])
        mask = obj > conf_thresh
        if mask.sum() == 0:
            continue
        idxs = mask.nonzero(as_tuple=False)
        for gj, gi in idxs.tolist():
            score = obj[gj, gi].item()
            tx, ty, tw, th = pred[gj, gi, 1:5].tolist()
            cx, cy = (gi + tx) * cell, (gj + ty) * cell
            w, h = math.exp(tw) * anchor_w, math.exp(th) * anchor_h
            all_boxes.append([cx - w / 2, cy - h / 2, w, h])
            all_scores.append(score)

    if not all_boxes:
        return [], []
    all_boxes, all_scores = np.array(all_boxes), np.array(all_scores)
    keep = nms(all_boxes, all_scores, iou_thresh=0.45)
    return all_boxes[keep].tolist(), all_scores[keep].tolist()


# ---------------- the thing an app actually calls ----------------

class ProductDetectorRuntime:
    """
    Loads the trained model + config once, then lets you run inference on
    any image path or PIL image, as many times as you want.
    """

    def __init__(self, ckpt_path="models/model_best.pth", config_path="config_files/config.json", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(config_path, "r") as f:
            cfg = json.load(f)
        self.img_size = cfg["img_size"]
        self.grid_sizes = cfg["grid_sizes"]
        self.anchors = cfg["anchors"]

        self.model = ProductDetector(pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def predict(self, image, conf_thresh=0.3):
        """
        image: file path (str) or a PIL.Image
        returns: (boxes, scores) - boxes are [x, y, w, h] in resized (img_size x img_size) coordinates
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        img_r = image.resize((self.img_size, self.img_size))

        arr = np.asarray(img_r, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)

        preds = self.model(tensor)
        preds = [p[0].cpu() for p in preds]
        boxes, scores = decode(preds, self.anchors, self.grid_sizes, self.img_size, conf_thresh)
        return boxes, scores

    def predict_and_draw(self, image, conf_thresh=0.3, color="lime"):
        """returns a PIL image (resized) with boxes drawn on it, and the count of products found"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        boxes, scores = self.predict(image, conf_thresh)

        vis = image.resize((self.img_size, self.img_size)).copy()
        draw = ImageDraw.Draw(vis)
        for (x, y, w, h) in boxes:
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        return vis, len(boxes)
