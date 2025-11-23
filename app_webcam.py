import cv2
import time

from utils.denoise import bilateral_denoise, anisotropic_diffusion_gray
from utils.style_transfer import load_transformer, stylize_frame
from utils.segmentation import load_person_segmenter, get_person_mask


def apply_anisotropic_color(frame_bgr):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    Lf = L.astype("float32") / 255.0
    Ld = anisotropic_diffusion_gray(Lf, n_iter=10, k=15, lam=0.25)
    L_out = (Ld * 255.0).astype("uint8")

    lab_out = cv2.merge([L_out, A, B])
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)


def main():
    styles = {
        "1": ("Mosaic",      load_transformer("weights/mosaic.pth")),
        "2": ("Picasso",     load_transformer("weights/picasso.pth")),
        "3": ("Candy",       load_transformer("weights/candy.pth")),
        "4": ("Starry",      load_transformer("weights/starry.pth")),
        "5": ("Wave",        load_transformer("weights/wave.pth")),
        "6": ("Udnie",       load_transformer("weights/udnie.pth")),
        "7": ("Lazy",        load_transformer("weights/lazy.pth")),
        "8": ("Tokyo Ghoul", load_transformer("weights/tokyo_ghoul.pth")),
    }

    current_key = "1"
    current_name, current_model = styles[current_key]

    denoise_mode = "none"
    region_mode = "full"

    seg_model, seg_preprocess = load_person_segmenter()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    prev_time = time.time()
    fps = 0.0

    print("Controls:")
    print("  1..8    - change style")
    print("  d       - cycle denoising: none -> bilateral -> anisotropic")
    print("  f       - stylize foreground (person) only")
    print("  b       - stylize background only")
    print("  a       - stylize all (full frame)")
    print("  q       - quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        max_side = max(h, w)
        scale = min(360 / max_side, 1.0)

        if scale < 1.0:
            frame_small = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            frame_small = frame

        if denoise_mode == "bilateral":
            pre = bilateral_denoise(frame_small)
        elif denoise_mode == "anisotropic":
            pre = apply_anisotropic_color(frame_small)
        else:
            pre = frame_small

        styled_small = stylize_frame(pre, current_model)

        if region_mode != "full":
            mask = get_person_mask(pre, seg_model, seg_preprocess)
            mask_3 = mask[:, :, None]

            pre_f = pre.astype("float32")
            styled_f = styled_small.astype("float32")

            if region_mode == "fg":
                mixed = styled_f * mask_3 + pre_f * (1.0 - mask_3)
            else:
                mixed = styled_f * (1.0 - mask_3) + pre_f * mask_3

            styled_small = mixed.clip(0, 255).astype("uint8")

        if scale < 1.0:
            styled = cv2.resize(
                styled_small, (w, h),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            styled = styled_small

        styled = styled.copy()

        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

        text1 = f"Style: {current_name} (keys 1..8)"
        text2 = f"Denoise: {denoise_mode} (key d)"
        text3 = f"Region: {region_mode} (a=all, f=fg, b=bg)"
        text4 = f"FPS: {fps:.1f}"

        y0 = 25
        dy = 25
        for i, txt in enumerate([text1, text2, text3, text4]):
            y = y0 + i * dy
            cv2.putText(
                styled, txt, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 3, cv2.LINE_AA,
            )
            cv2.putText(
                styled, txt, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 1, cv2.LINE_AA,
            )

        cv2.imshow("Real-time Style Transfer", styled)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("1"), ord("2"), ord("3"), ord("4"),
                     ord("5"), ord("6"), ord("7"), ord("8")):
            current_key = chr(key)
            current_name, current_model = styles[current_key]
        elif key == ord("d"):
            if denoise_mode == "none":
                denoise_mode = "bilateral"
            elif denoise_mode == "bilateral":
                denoise_mode = "anisotropic"
            else:
                denoise_mode = "none"
        elif key == ord("f"):
            region_mode = "fg"
        elif key == ord("b"):
            region_mode = "bg"
        elif key == ord("a"):
            region_mode = "full"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
