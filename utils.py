import math
import torch
import numpy as np
from PIL import Image

LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def comfy_images_to_pil(tensor):
    """Convert ComfyUI image tensor [B,H,W,C] float32 [0,1] to list of PIL RGB images."""
    images = []
    for i in range(tensor.shape[0]):
        img_np = (tensor[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        images.append(Image.fromarray(img_np, "RGB"))
    return images


def pil_to_comfy_images(images):
    """Convert list of PIL images to ComfyUI tensor [B,H,W,C] float32 [0,1]."""
    tensors = []
    for img in images:
        img = img.convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(img_np))
    return torch.stack(tensors)


def auto_resolution(width, height, target_pixels=768 * 768):
    """Preserve aspect ratio targeting ~1024x1024 total pixels, snapped to 8px."""
    if width == 0 or height == 0:
        return 1024, 1024
    aspect = width / height
    new_height = math.sqrt(target_pixels / aspect)
    new_width = new_height * aspect
    new_width = max(8, round(new_width / 8) * 8)
    new_height = max(8, round(new_height / 8) * 8)
    return new_width, new_height


def make_step_callback(progress_bar):
    """Return a diffusers callback_on_step_end that advances a ComfyUI progress bar."""
    def callback(pipe, step_index, timestep, callback_kwargs):
        progress_bar.update(1)
        return callback_kwargs
    return callback
