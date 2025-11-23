# Image Style Transfer App (CS663 Project)

Neural style transfer lets us repaint an image or a video in the style of an artwork using convolutional neural networks (CNNs).  

This project combines **two complementary systems**:

1. **Offline Gatys-style optimization** for **high-quality image and video stylization**.
2. **Real-time feed-forward style transfer** with **denoising** and **region-aware effects** (foreground/background control) via a desktop OpenCV app and a Streamlit web app.

This repository contains all the code used for the CS663 (Fundamentals of Digital Image Processing) project at IIT Bombay.

---

## Features

### 1. Gatys-Based Optimization Style Transfer

- Implements the original **Gatys et al.** optimization-based style transfer using **VGG19**.
- Works on:
  - **Individual images**
  - **Videos** (frame extraction → per-frame optimization → recombination)
- Objective:
  \[
    \mathcal{L}(x) =
    \lambda_c \mathcal{L}_{\text{content}}(x , c)
    + \lambda_s \mathcal{L}_{\text{style}}(x , s)
    + \lambda_{tv} \mathcal{L}_{\text{TV}}(x)
  \]
- Uses:
  - Content loss on VGG feature maps
  - Style loss via Gram matrices
  - Total Variation (TV) regularization for smoothness
- Produces **painterly, high-detail stylization** suitable for short clips and high-quality renders.

### 2. Real-Time Feed-Forward Style Transfer

- Uses **pretrained feed-forward networks** for fast style transfer:
  - A custom **TransformerNet** with depthwise separable convolutions.
  - A **Johnson-style network** (encoder–residual–decoder).
- Supports **8 artistic styles**:
  - **TransformerNet**:
    - `mosaic.pth` – blocky, textured strokes
    - `picasso.pth` – cubist, angular color fragments
    - `candy.pth` – pastel swirls and smooth textures
  - **JohnsonNet**:
    - `starry.pth` – Van Gogh “Starry Night” style
    - `wave.pth` – Hokusai-style wave textures
    - `udnie.pth` – strong abstract distortions
    - `lazy.pth` – thick brush strokes, painterly smoothing
    - `tokyo_ghoul.pth` – dark anime-style shading

- Runs in:
  - **Desktop webcam app** using OpenCV (`app_webcam.py`)
  - **Streamlit web app** for images / webcam snapshots (`web_app.py`)

### 3. Region-Aware Stylization (Foreground / Background)

- Uses **DeepLabV3 ResNet50** (from `torchvision.models.segmentation`) for **person segmentation**.
- Generates a per-pixel **person mask** and combines it with stylized output:
  - **Foreground-only** stylization (only the person is stylized)
  - **Background-only** stylization (only background is stylized)
  - **Full-frame** stylization
- Blending:
  \[
    O = S \cdot M + I \cdot (1 - M) \quad\text{(foreground only)}
  \]
  \[
    O = S \cdot (1 - M) + I \cdot M \quad\text{(background only)}
  \]

### 4. Denoising Before Stylization

To avoid amplifying sensor noise or compression artifacts, the pipeline includes optional pre-style-transfer denoising:

- **Bilateral filter**
  - Edge-preserving smoothing with spatial + range kernels
- **Anisotropic diffusion (Perona–Malik)**
  - Diffusion on L channel in LAB space for structure-preserving smoothing

Denoising is implemented in `utils/denoise.py` and can be toggled at runtime.

---

## Repository Structure

```text
Image-Style-Transfer-App/
├─ .devcontainer/           # Dev container config (VS Code / remote dev)
├─ models/                  # Style transfer network architectures
│  └─ transformer_net.py
├─ utils/
│  ├─ style_transfer.py     # Core stylization functions for real-time app
│  ├─ denoise.py            # Bilateral filter and anisotropic diffusion
│  └─ segmentation.py       # DeepLabV3 loading and person mask generation
├─ weights/                 # Pretrained style network weights (.pth files)
├─ app_webcam.py            # Real-time webcam style transfer application
├─ web_app.py               # Streamlit app for images / webcam snapshots
├─ style-transfer.ipynb     # Image style transfer (Gatys/experiments)
├─ style-transfer-video.ipynb # Video style transfer pipeline (frames+video)
├─ requirements.txt         # Python dependencies
└─ README.md                # You are here
