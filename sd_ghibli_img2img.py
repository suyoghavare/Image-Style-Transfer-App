import os
from pathlib import Path

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image


# pick device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_ghibli_pipeline(
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
    lora_path: str | None = None,
):
    from diffusers import StableDiffusionImg2ImgPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )

    # memory saving for 4 GB 3050
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    pipe = pipe.to(device)

    if lora_path is not None:
        print(f"Loading LoRA from: {lora_path}")
        # new diffusers API uses PEFT under the hood
        pipe.load_lora_weights(lora_path, adapter_name="ghibli")
        pipe.set_adapters("ghibli")

    return pipe




def run_ghibli_img2img(
    input_path: str,
    output_path: str,
    lora_path: str,
    prompt: str = "same composition, same person, studio ghibli painting style,"
    "soft colors, clean lines, anime background",
    negative_prompt: str = "change of pose, different character, extra limbs, low quality, blurry, distorted",
    strength: float = 0.4,          # how much to change the image (0.3–0.8)
    guidance_scale: float = 10,    # how strongly prompt is followed (5–12)
    num_inference_steps: int = 1000,  # more steps = better quality but slower
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    pipe = load_ghibli_pipeline(lora_path=lora_path)

    init_image = Image.open(input_path).convert("RGB")

    # keep aspect ratio, fit inside 512x512 with padding
    target = 512
    w, h = init_image.size
    scale = min(target / w, target / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = init_image.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (target, target), (255, 255, 255))
    x0 = (target - new_w) // 2
    y0 = (target - new_h) // 2
    canvas.paste(resized, (x0, y0))
    init_image = canvas


    # create output folder
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # inference
    generator = torch.Generator(device=DEVICE).manual_seed(42)

    with torch.autocast(DEVICE if DEVICE == "cuda" else "cpu"):
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

    out_img = result.images[0]
    out_img.save(output_path)
    print(f"Saved Ghibli-style image to: {output_path}")


if __name__ == "__main__":
    # adjust these paths to your setup
    INPUT_IMG = "inputs/OIP.jpg"        # your original image
    OUTPUT_IMG = "outputs/test1_ghibli.png"
    LORA_PATH = "loras/ghibli_v2-03.safetensors"

    run_ghibli_img2img(
        input_path=INPUT_IMG,
        output_path=OUTPUT_IMG,
        lora_path=LORA_PATH,
    )



