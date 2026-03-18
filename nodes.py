import os
import gc
import torch

import comfy.model_management as mm
import comfy.utils
import folder_paths

from .utils import comfy_images_to_pil, pil_to_comfy_images, auto_resolution, make_step_callback

# ---------------------------------------------------------------------------
# Pipeline cache
# ---------------------------------------------------------------------------
_cached_pipe = None
_cached_key = None

DEFAULT_NEGATIVE = (
    "worst quality, low quality, bad anatomy, bad hands, missing fingers, "
    "extra fingers, blurry, watermark, text, signature, deformed, ugly"
)

BASE_MODEL_ID = "FireRedTeam/FireRed-Image-Edit-1.1"
FAST_TRANSFORMER_ID = "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19"


def _resolve_or_download(repo_id, models_dir):
    """Return a local path for repo_id, downloading via snapshot_download if needed."""
    if os.path.isdir(repo_id):
        return repo_id
    safe_name = repo_id.replace("/", "--")
    local_path = os.path.join(models_dir, safe_name)
    if not os.path.exists(local_path):
        print(f"[FireRedFast] Downloading {repo_id} → {local_path} ...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=local_path, local_dir_use_symlinks=False)
        print(f"[FireRedFast] Download complete: {repo_id}")
    return local_path


# ---------------------------------------------------------------------------
# Loader node
# ---------------------------------------------------------------------------

class FireRedFastLoader:
    """
    Loads the FireRed-Image-Edit-1.1 pipeline with the fast Qwen transformer
    (prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19).  Optionally tries to enable
    Flash-Attention-3 for extra speed.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_model_id": ("STRING", {
                    "default": BASE_MODEL_ID,
                    "tooltip": "HuggingFace repo ID or local path for the base pipeline",
                }),
                "fast_transformer_id": ("STRING", {
                    "default": FAST_TRANSFORMER_ID,
                    "tooltip": "HuggingFace repo ID or local path for the fast transformer weights",
                }),
                "precision": (["bf16", "fp16"], {"default": "bf16"}),
                "offload": (
                    ["model_cpu_offload", "sequential_cpu_offload", "full_gpu"],
                    {"default": "model_cpu_offload"},
                ),
                "enable_fa3": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Try to enable Flash Attention 3 (requires fa3 / flash-attn installed)",
                }),
            },
        }

    RETURN_TYPES = ("FIRERED_FAST_PIPE",)
    RETURN_NAMES = ("firered_fast_pipe",)
    FUNCTION = "load_pipeline"
    CATEGORY = "FireRedEdit/Fast"
    DESCRIPTION = (
        "Loads FireRed-Image-Edit-1.1 with the fast Rapid-AIO transformer for ~4-step editing. "
        "The pipeline is cached between runs so reloading is instant when settings are unchanged."
    )

    def load_pipeline(self, base_model_id, fast_transformer_id, precision, offload, enable_fa3):
        global _cached_pipe, _cached_key

        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        cache_key = (base_model_id, fast_transformer_id, precision, offload, enable_fa3)

        if _cached_pipe is not None and _cached_key == cache_key:
            print("[FireRedFast] Using cached pipeline.")
            return ({"pipeline": _cached_pipe, "dtype": dtype},)

        # Clear previous pipeline
        if _cached_pipe is not None:
            del _cached_pipe
            _cached_pipe = None
            _cached_key = None
            gc.collect()

        mm.unload_all_models()
        mm.soft_empty_cache()

        models_dir = os.path.join(folder_paths.models_dir, "diffusers")
        os.makedirs(models_dir, exist_ok=True)

        pbar = comfy.utils.ProgressBar(4)

        # Resolve / download models
        base_path = _resolve_or_download(base_model_id, models_dir)
        pbar.update(1)
        transformer_path = _resolve_or_download(fast_transformer_id, models_dir)
        pbar.update(1)

        # Import the bundled pipeline classes (local qwenimage module)
        from .qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
        from .qwenimage.transformer_qwenimage import QwenImageTransformer2DModel

        device = mm.get_torch_device()

        print(f"[FireRedFast] Loading fast transformer from {transformer_path} ...")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            transformer_path,
            torch_dtype=dtype,
        )

        print(f"[FireRedFast] Loading pipeline from {base_path} ({precision}) ...")
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            base_path,
            transformer=transformer,
            torch_dtype=dtype,
        )
        pbar.update(1)

        # Flash Attention 3
        if enable_fa3:
            try:
                from .qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3
                pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
                print("[FireRedFast] Flash Attention 3 enabled.")
            except Exception as e:
                print(f"[FireRedFast] Could not enable FA3: {e}")

        # Apply offload / device strategy
        if offload == "model_cpu_offload":
            pipe.enable_model_cpu_offload()
        elif offload == "sequential_cpu_offload":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(device)

        pbar.update(1)

        _cached_pipe = pipe
        _cached_key = cache_key

        print(f"[FireRedFast] Pipeline ready.")
        return ({"pipeline": pipe, "dtype": dtype},)


# ---------------------------------------------------------------------------
# Sampler node
# ---------------------------------------------------------------------------

class FireRedFastSampler:
    """
    Runs image-editing inference with the FireRed-Image-Edit-1.1 / fast pipeline.
    Connect one or two images and describe the edit in the prompt.
    When two images are connected they are referenced as 'Picture 1' and 'Picture 2'
    inside the model — useful for style/clothing/glasses transfer tasks.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "firered_fast_pipe": ("FIRERED_FAST_PIPE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Change the style to oil painting.",
                    "tooltip": "Editing instruction. Use 'Picture 1' / 'Picture 2' to reference inputs.",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "steps": ("INT", {
                    "default": 4, "min": 1, "max": 50,
                    "tooltip": "4 steps is the intended fast mode; increase for higher quality.",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1,
                    "tooltip": "true_cfg_scale. 1.0 works best for the fast distilled model.",
                }),
            },
            "optional": {
                "image1": ("IMAGE", {"tooltip": "Primary image to edit (Picture 1)"}),
                "image2": ("IMAGE", {"tooltip": "Secondary reference image (Picture 2)"}),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_NEGATIVE,
                }),
                "width": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 8,
                    "tooltip": "0 = auto from image1 aspect ratio",
                }),
                "height": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 8,
                    "tooltip": "0 = auto from image1 aspect ratio",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "FireRedEdit/Fast"
    DESCRIPTION = (
        "Edits images with the FireRed-Image-Edit-1.1 fast pipeline. "
        "Connect image1 (and optionally image2) then describe your edit. "
        "Default 4 steps is sufficient for the fast model."
    )

    def generate(
        self,
        firered_fast_pipe,
        prompt,
        seed,
        steps,
        guidance_scale,
        image1=None,
        image2=None,
        negative_prompt=DEFAULT_NEGATIVE,
        width=0,
        height=0,
    ):
        pipe = firered_fast_pipe["pipeline"]

        # Collect PIL images
        pil_images = []
        for img_tensor in [image1, image2]:
            if img_tensor is not None:
                pil_images.append(comfy_images_to_pil(img_tensor)[0])

        # Resolve output dimensions
        if pil_images:
            if width == 0 or height == 0:
                width, height = auto_resolution(pil_images[0].width, pil_images[0].height)
            image_arg = pil_images if len(pil_images) > 1 else pil_images[0]
        else:
            # Text-to-image fallback — provide a blank image
            if width == 0:
                width = 1024
            if height == 0:
                height = 1024
            from PIL import Image as PILImage
            image_arg = PILImage.new("RGB", (width, height), (255, 255, 255))

        generator = torch.Generator(device="cpu").manual_seed(seed)
        pbar = comfy.utils.ProgressBar(steps)
        callback = make_step_callback(pbar)

        print(
            f"[FireRedFast] Generating {width}x{height}, steps={steps}, "
            f"cfg={guidance_scale}, images={len(pil_images)}, seed={seed}"
        )

        with torch.no_grad():
            result = pipe(
                image=image_arg,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                true_cfg_scale=guidance_scale,
                generator=generator,
                callback_on_step_end=callback,
            )

        return (pil_to_comfy_images(result.images),)


# ---------------------------------------------------------------------------
# Unloader node
# ---------------------------------------------------------------------------

class FireRedFastUnloader:
    """Explicitly frees the pipeline from memory."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"firered_fast_pipe": ("FIRERED_FAST_PIPE",)}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "unload"
    CATEGORY = "FireRedEdit/Fast"
    DESCRIPTION = "Releases the FireRed-Fast pipeline from GPU/CPU memory."

    def unload(self, firered_fast_pipe):
        global _cached_pipe, _cached_key
        pipe = firered_fast_pipe.get("pipeline")
        try:
            pipe.remove_all_hooks()
        except Exception:
            pass
        del pipe
        del firered_fast_pipe
        _cached_pipe = None
        _cached_key = None
        gc.collect()
        mm.soft_empty_cache()
        print("[FireRedFast] Pipeline unloaded.")
        return ()


# ---------------------------------------------------------------------------
# Registrations
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "FireRedFastLoader": FireRedFastLoader,
    "FireRedFastSampler": FireRedFastSampler,
    "FireRedFastUnloader": FireRedFastUnloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FireRedFastLoader": "FireRed Fast Loader (1.1)",
    "FireRedFastSampler": "FireRed Fast Sampler (1.1)",
    "FireRedFastUnloader": "FireRed Fast Unloader (1.1)",
}
