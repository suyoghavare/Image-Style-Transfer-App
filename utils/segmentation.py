import cv2
import numpy as np
import torch
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PERSON_CLASS = 15


def load_person_segmenter():
   
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights).to(device).eval()
    preprocess = weights.transforms()
    return model, preprocess


def get_person_mask(frame_bgr, model, preprocess, thresh: float = 0.5):

    h, w, _ = frame_bgr.shape

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)

    inp = preprocess(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(inp)["out"][0]          # (C, H', W')
        probs = torch.softmax(logits, dim=0)   # class probabilities
        person = probs[PERSON_CLASS]           # (H', W')
        mask_small = (person > thresh).float()

    mask_small = mask_small.cpu().numpy().astype("float32")
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask
