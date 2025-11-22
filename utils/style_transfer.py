import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_transformer(model_path: str):
    """
    Load style network from .pth.

    - TransformerNet (mosaic / picasso / candy)
      keys like "conv1.weight", "in1.weight", ...
    - JohnsonNet (udnie / wave / starry / lazy / tokyo_ghoul)
      keys like "ConvBlock.*", "ResidualBlock.*"
    """
    from models.transformer_net import TransformerNet, JohnsonNet

    state = torch.load(model_path, map_location=device)

    keys = list(state.keys())
    if any(k.startswith("ConvBlock.") for k in keys):
        model = JohnsonNet().to(device)
    else:
        model = TransformerNet().to(device)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def bgr_to_tensor(frame_bgr):
    frame_rgb = frame_bgr[:, :, ::-1]
    img = Image.fromarray(frame_rgb)
    transform = transforms.Compose([
        transforms.ToTensor()  # [0,1]
    ])
    tensor = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]
    return tensor


def tensor_to_bgr(tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    img = tensor[0].permute(1, 2, 0).numpy()  # HWC RGB
    img = (img * 255.0).astype("uint8")
    frame_rgb = img
    frame_bgr = frame_rgb[:, :, ::-1].copy()
    return frame_bgr


def stylize_frame(frame_bgr, model, alpha=0.8):
    """
    frame_bgr  uint8 OpenCV frame
    model      TransformerNet
    alpha      style strength [0,1]
    """
    with torch.no_grad():
        x = bgr_to_tensor(frame_bgr)      # [1,3,H,W] in [0,1]
        y = model(x)                      # arbitrary range

        # match spatial size just in case
        if y.shape[2:] != x.shape[2:]:
            y = F.interpolate(
                y,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # per frame min max normalize to [0,1]
        y_min = y.amin(dim=[1, 2, 3], keepdim=True)
        y_max = y.amax(dim=[1, 2, 3], keepdim=True)
        y_norm = (y - y_min) / (y_max - y_min + 1e-5)

        # blend with original for stability
        y_blend = alpha * y_norm + (1.0 - alpha) * x

        out_bgr = tensor_to_bgr(y_blend)
    return out_bgr
