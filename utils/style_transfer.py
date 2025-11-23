import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_transformer(model_path: str):

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


_to_tensor = transforms.Compose([transforms.ToTensor()])


def bgr_to_tensor(frame_bgr):
    rgb = frame_bgr[:, :, ::-1]
    img = Image.fromarray(rgb)
    tensor = _to_tensor(img).unsqueeze(0).to(device)
    return tensor


def tensor_to_bgr(tensor):
    t = tensor.detach().cpu().clamp(0, 1)
    img = t[0].permute(1, 2, 0).numpy()
    img = (img * 255.0).astype("uint8")
    rgb = img
    bgr = rgb[:, :, ::-1].copy()
    return bgr


def stylize_frame(frame_bgr, model, alpha: float = 0.8):

    with torch.no_grad():
        x = bgr_to_tensor(frame_bgr)

        y = model(x)

        if y.shape[2:] != x.shape[2:]:
            y = F.interpolate(
                y,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        y_min = y.amin(dim=[1, 2, 3], keepdim=True)
        y_max = y.amax(dim=[1, 2, 3], keepdim=True)
        y_norm = (y - y_min) / (y_max - y_min + 1e-5)

        y_blend = alpha * y_norm + (1.0 - alpha) * x

        return tensor_to_bgr(y_blend)
