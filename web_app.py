import io
import time

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from utils.style_transfer import load_transformer, stylize_frame
from utils.denoise import bilateral_denoise, anisotropic_diffusion_gray
from utils.segmentation import load_person_segmenter, get_person_mask


# Simple helper
def pil_to_bgr(pil_img):
    rgb = np.array(pil_img.convert("RGB"))
    bgr = rgb[:, :, ::-1].copy()
    return bgr


def bgr_to_pil(bgr_img):
    rgb = bgr_img[:, :, ::-1]
    return Image.fromarray(rgb)


def apply_anisotropic_color(frame_bgr):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    Lf = L.astype("float32") / 255.0
    Ld = anisotropic_diffusion_gray(Lf, n_iter=10, k=15, lam=0.25)
    L_out = (Ld * 255.0).astype("uint8")

    lab_out = cv2.merge([L_out, A, B])
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)


@st.cache_resource
def load_all_styles():
    styles = {
        "Mosaic":       load_transformer("weights/mosaic.pth"),
        "Picasso":      load_transformer("weights/picasso.pth"),
        "Candy":        load_transformer("weights/candy.pth"),
        "Starry":       load_transformer("weights/starry.pth"),
        "Wave":         load_transformer("weights/wave.pth"),
        "Udnie":        load_transformer("weights/udnie.pth"),
        "Lazy":         load_transformer("weights/lazy.pth"),
        "Tokyo Ghoul":  load_transformer("weights/tokyo_ghoul.pth"),
    }
    return styles



@st.cache_resource
def load_segmenter_cached():
    return load_person_segmenter()  # returns (model, preprocess)


st.set_page_config(
    page_title="CS663 Style Transfer Demo",
    layout="wide"
)

st.title("Real Time Style Transfer Web")

st.markdown(
    "Upload an image or capture from camera, "
    "choose a style and optionally apply it only on foreground or background."
)

# Sidebar controls
with st.sidebar:
    st.header("Controls")

    style_name = st.selectbox(
        "Style",
        ["Mosaic", "Picasso", "Candy",
         "Starry", "Wave", "Udnie", "Lazy", "Tokyo Ghoul"],
        index=0,
    )


    region_mode = st.radio(
        "Region to stylize",
        ["Full image", "Foreground only", "Background only"],
        index=0,
    )

    denoise_mode = st.radio(
        "Denoising",
        ["None", "Bilateral", "Anisotropic"],
        index=0,
    )

    alpha = st.slider(
        "Style strength",
        min_value=0.3,
        max_value=1.0,
        value=0.8,
        step=0.05,
    )

    max_side = st.slider(
        "Max processing size (pixels)",
        min_value=256,
        max_value=720,
        value=480,
        step=64,
        help="Smaller is faster"
    )

    st.markdown("---")
    st.caption("Made with love by Suyog & Karan ❤️")


# Main area: input choice
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Input image")

    input_mode = st.radio(
        "Choose input source",
        ["Upload image", "Use webcam"],
        index=0,
        horizontal=True,
    )

    uploaded_image = None

    if input_mode == "Upload image":
        file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if file is not None:
            uploaded_image = Image.open(io.BytesIO(file.read())).convert("RGB")
    else:
        cam_img = st.camera_input("Capture from webcam")
        if cam_img is not None:
            uploaded_image = Image.open(cam_img).convert("RGB")

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Original image", use_container_width=True)


with col_right:
    st.subheader("Stylized output")
    output_placeholder = st.empty()
    info_placeholder = st.empty()
    download_placeholder = st.empty()

    if uploaded_image is None:
        output_placeholder.info("Upload or capture an image to see the result.")
    else:
        # Load models
        styles = load_all_styles()
        model = styles[style_name]

        seg_model, seg_preprocess = load_segmenter_cached()

        # Convert to BGR
        frame_bgr = pil_to_bgr(uploaded_image)

        # Resize for speed
        h, w, _ = frame_bgr.shape
        scale = min(max_side / max(h, w), 1.0)
        if scale < 1.0:
            frame_small = cv2.resize(
                frame_bgr,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            frame_small = frame_bgr

        # Denoise
        if denoise_mode == "Bilateral":
            pre = bilateral_denoise(frame_small)
        elif denoise_mode == "Anisotropic":
            pre = apply_anisotropic_color(frame_small)
        else:
            pre = frame_small

        start_time = time.time()

        # Full frame stylization
        styled_small = stylize_frame(pre, model, alpha=alpha)

        # Region aware blending
        if region_mode != "Full image":
            mask = get_person_mask(pre, seg_model, seg_preprocess)  # [H,W]
            mask_3 = mask[:, :, None]  # [H,W,1]

            pre_f = pre.astype("float32")
            styled_f = styled_small.astype("float32")

            if region_mode == "Foreground only":
                mixed = styled_f * mask_3 + pre_f * (1.0 - mask_3)
            else:  # Background only
                mixed = styled_f * (1.0 - mask_3) + pre_f * mask_3

            styled_small = mixed.clip(0, 255).astype("uint8")

        # Upscale back
        if scale < 1.0:
            styled = cv2.resize(
                styled_small,
                (w, h),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            styled = styled_small

        elapsed = time.time() - start_time

        out_pil = bgr_to_pil(styled)

        # show image
        output_placeholder.image(
            out_pil,
            caption=f"{style_name} result",
            use_container_width=True,
        )
        info_placeholder.caption(f"Processing time: {elapsed:.2f} s")

        # prepare download
        buf = io.BytesIO()
        out_pil.save(buf, format="PNG")
        buf.seek(0)

        download_placeholder.download_button(
            label="Download stylized image",
            data=buf,
            file_name=f"{style_name.lower()}_styled.png",
            mime="image/png",
        )

