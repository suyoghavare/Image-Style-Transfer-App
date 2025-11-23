# Image Style Transfer App
CS663 – Fundamentals of Digital Image Processing  
Indian Institute of Technology Bombay  
Authors: Suyog Havare (25M1061), Karan Patel (25M1090)

---

## Introduction

This repository implements a complete image and video style transfer system using convolutional neural networks. The project includes:

1. An optimization-based implementation of the Gatys et al. neural style transfer algorithm for both images and videos.
2. A real-time feed-forward style transfer system using pretrained networks.
3. Region-aware stylization using DeepLabV3 person segmentation.
4. Denoising modules (bilateral filter and anisotropic diffusion) to improve visual quality.
5. Two user interfaces: an OpenCV webcam app and a Streamlit web app.

The goal is to combine classical digital image processing with deep learning in a unified pipeline.

---

## Gatys Optimization-Based Style Transfer (Images + Video)

We implement the original optimization-based approach by Gatys et al. using VGG19 features.

Given content frame c, style image s, and output image x, the objective is:

Total loss:
L(x) = lc * L_content(x, c) + ls * L_style(x, s) + ltv * L_TV(x)

Content loss (at VGG layer lc):
L_content = ||F_lc(x) - F_lc(c)||^2

Style loss (using Gram matrices):
G_l(x) = (1 / (H * W)) * F_l(x) * F_l(x)^T
L_style = (1 / |S|) * sum over l in S of ||G_l(x) - G_l(s)||^2

Total variation regularization:
L_TV = ||x[:,:,:,:-1] - x[:,:,:,1:]||_1  +  ||x[:,:,:-1,:] - x[:,:,1:,:]||_1

For videos, the pipeline is:
1. Extract frames from the video.
2. Apply the Gatys optimization to each frame.
3. Reconstruct the stylized video.

Video demo:
https://www.youtube.com/shorts/m4p21UDbUJY

---

## Feed-Forward Real-Time Style Transfer

Fast style transfer is implemented using pretrained feed-forward networks:

- TransformerNet (lightweight, depthwise separable)
- JohnsonNet (classic fast style transfer)

Each style has its own generator network. Complexity is linear in image size, making it suitable for real-time video.

Supported styles (8 total):

TransformerNet:
- mosaic.pth
- picasso.pth
- candy.pth

JohnsonNet:
- starry.pth
- wave.pth
- udnie.pth
- lazy.pth
- tokyo_ghoul.pth

Real-time demo:
https://youtu.be/WdWywhPwFkU

---

## Region-Aware Stylization (Foreground/Background Control)

We use DeepLabV3-ResNet50 (from torchvision) to generate a person mask M.

Given:
- I: original frame
- S: stylized frame
- M: mask (1 for person, 0 for background)

Foreground-only:
O = S * M + I * (1 - M)

Background-only:
O = S * (1 - M) + I * M

Full-frame:
O = S

Both the webcam app and Streamlit app support this feature.

---

## Denoising Module

Noise distorts stylization, especially in textured styles. Two denoising methods are included:

1. Bilateral filter  
   Uses spatial and range kernels for edge-preserving smoothing.

2. Anisotropic diffusion  
   Based on the Perona-Malik equation.  
   Diffusion is applied only on the L-channel of LAB space.

Users can toggle denoising modes at runtime.

---

## System Architecture

Shared core modules:

- models/transformer_net.py
- utils/style_transfer.py
- utils/denoise.py
- utils/segmentation.py

Two front-end applications:

1. Real-time webcam interface (OpenCV)
2. Streamlit web application (images or webcam snapshot)

Both applications use the same:
- style networks
- denoising system
- segmentation system

---

## Webcam Application (OpenCV)

Implemented in `app_webcam.py`.

Features:
- Real-time style transfer
- Keyboard controls:
  - 1–8: select style
  - d: change denoise mode
  - a: full-frame stylization
  - f: foreground-only
  - b: background-only
  - q: quit

Demo video:
https://youtu.be/WdWywhPwFkU

---

## Streamlit Application

Implemented in `web_app.py`.

Features:
- Upload image or capture webcam snapshot
- Choose style and style strength
- Choose region mode (full/foreground/background)
- Choose denoise mode
- Processing time display
- Download stylized output

Run using:
python -m streamlit run web_app.py

Screenshots are provided in the presentation PDF.

---

## Results Summary

1. Gatys-based still images  
   High-quality results with detailed textures.

2. Gatys-based videos  
   Frame-by-frame optimization produces painterly animations.

3. Feed-forward models  
   Eight styles with distinct artistic characteristics.

4. Region-awareness  
   Flexible control of stylization regions.

5. Denoising  
   Reduces noise amplification significantly.

---

## Limitations and Future Work

- DeepLabV3 is heavy and adds latency.
- Masks may fail under occlusion or strong pose changes.
- Some styles introduce strong distortions.

Future improvements:
- Lighter segmentation models
- Multi-style single network
- Temporal consistency for video

---

## References and Resources

Gatys et al. Original Paper:  
https://arxiv.org/pdf/1508.06576

Johnson et al. Fast Style Transfer Paper:  
https://arxiv.org/pdf/1603.08155

Pretrained style models (MobileNetV2-based):  
https://github.com/mmalotin/pytorch-fast-neural-style-mobilenetV2/tree/master/models

Additional style weight (Udnie):  
https://github.com/rrmina/fast-neural-style-pytorch/blob/master/transforms/udnie.pth

Project Google Drive:  
https://drive.google.com/drive/folders/1fL4PVdlk9nVbPRUSgfi3NFebMX5O_0Hm?usp=sharing

Kaggle reference notebook (video stylization):  
https://www.kaggle.com/code/kxrxn0804/style-transfer-video

Project Repository:  
https://github.com/suyoghavare/Image-Style-Transfer-App
