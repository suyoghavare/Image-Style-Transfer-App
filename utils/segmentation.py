# utils/segmentation.py

import cv2
import numpy as np
import torch
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PERSON_CLASS = 15  # Pascal VOC label index for "person"


def load_person_segmenter():
    """
    Load DeepLabV3-ResNet50 pretrained on VOC classes.
    """
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights).to(device).eval()
    preprocess = weights.transforms()
    return model, preprocess


def get_person_mask(frame_bgr, model, preprocess, thresh: float = 0.5):
    """
    frame_bgr : uint8 (H, W, 3) BGR
    returns   : float32 mask (H, W) in [0,1], where 1≈person
    """
    h, w, _ = frame_bgr.shape

    # BGR -> RGB -> PIL
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)

    # torchvision weights.transforms() expects PIL image
    tensor = preprocess(img_pil).unsqueeze(0).to(device)  # [1,3,H',W']

    with torch.no_grad():
        out = model(tensor)["out"][0]  # [C,H',W']

        probs = torch.softmax(out, dim=0)
        person_prob = probs[PERSON_CLASS]  # [H',W']

        mask_small = (person_prob > thresh).float()  # 0/1

    # back to numpy and resize to original size
    mask_small = mask_small.cpu().numpy().astype("float32")
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask  # [H,W] float32 in {0,1}

