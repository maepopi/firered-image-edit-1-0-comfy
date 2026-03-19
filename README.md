# FireRed Image Edit 1.0 — ComfyUI Custom Node

A ComfyUI custom node for **FireRed-Image-Edit-1.1** with the fast **Qwen-Image-Edit-Rapid-AIO-V19** transformer. Edit images in ~4 steps using a vision-language model as the backbone.

---

## What this is

[FireRed-Image-Edit-1.1](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1) is an instruction-based image editing model built on top of Qwen2.5-VL. You describe what you want changed in plain language and the model applies the edit while preserving unrelated parts of the image.

The [Rapid-AIO transformer](https://huggingface.co/prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19) is a distilled version that runs in as few as 4 steps instead of the usual 20–50, making it practical for iterative editing workflows.

Original Space: [prithivMLmods/FireRed-Image-Edit-1.0-Fast](https://huggingface.co/spaces/prithivMLmods/FireRed-Image-Edit-1.0-Fast)

---

## System requirements

| Component | Minimum |
|-----------|---------|
| GPU VRAM  | 12 GB   |
| System RAM | 42 GB free at load time |
| Disk space | ~36 GB for model weights |

### Why so much RAM?

The fast transformer weights are stored on disk in **float8** format (20 GB). When loaded into PyTorch for inference they must be converted to **bfloat16** (2 bytes per parameter instead of 1), which doubles the in-memory size to ~40 GB. This conversion is unavoidable on GPUs that do not support native float8 compute (anything before Hopper/H100).

The text encoder (Qwen2.5-VL 7B) adds another ~16 GB on disk, but because it is memory-mapped with `low_cpu_mem_usage=True` its pages load lazily — it barely increases RAM usage at load time. Total settled RAM after loading: **~42 GB**.

### Why `sequential_cpu_offload`?

The combined model is ~56 GB in bfloat16, far larger than a 12 GB GPU. `sequential_cpu_offload` solves this by keeping all weights in CPU RAM and moving one layer at a time to the GPU just before its forward pass, then immediately moving it back. Peak VRAM during inference is ~1–2 GB per layer rather than the full model size.

Alternative approaches that were tried and abandoned during development:

- **`device_map="cuda"` for the transformer** — conflicts with `sequential_cpu_offload` hooks, causing both to fight over device placement and producing garbage output.
- **`disk_offload`** — streams weights from disk instead of RAM, which would drop settled RAM to ~9 GB. Abandoned because the `init_empty_weights` fast-load path (needed to avoid a 40 GB RAM spike on every session start) crashes for this custom model class. The slow path still hit the RAM limit when combined with ComfyUI's own memory usage.

`sequential_cpu_offload` is the only approach verified working end-to-end on a 12 GB VRAM / 62 GB RAM system.

---

## Installation

### 1. Clone into your ComfyUI custom nodes directory

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/maepopi/firered-image-edit-1-0-comfy firered-image-edit-1-0
```

### 2. Install dependencies

Activate your ComfyUI Python environment, then:

```bash
pip install -r firered-image-edit-1-0/requirements.txt
```

### 3. Models download automatically

On the first run the Loader node downloads both models from HuggingFace into `ComfyUI/models/diffusers/`:

- `FireRedTeam--FireRed-Image-Edit-1.1` (~54 GB total)
- `prithivMLmods--Qwen-Image-Edit-Rapid-AIO-V19` (~20 GB)

This only happens once. Subsequent runs verify and resume any incomplete downloads.

---

## Nodes

### FireRed Fast Loader (1.1)

Loads the pipeline and caches it. The cache persists for the entire ComfyUI session — rerunning the workflow with the same settings is instant.

| Parameter | Description |
|-----------|-------------|
| `base_model_id` | HuggingFace repo ID or local path for the base pipeline. Default: `FireRedTeam/FireRed-Image-Edit-1.1` |
| `fast_transformer_id` | HuggingFace repo ID or local path for the fast transformer. Default: `prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19` |
| `precision` | `bf16` (recommended) or `fp16`. `bf16` matches the original training dtype. |
| `enable_fa3` | Enable Flash Attention 3 for faster inference. Requires `flash-attn` installed separately. Leave off if unsure. |

**Important:** Before loading, the node calls `unload_all_models()` to free all other ComfyUI models from memory. Do not run other memory-heavy nodes in the same workflow as the loader.

---

### FireRed Fast Sampler (1.1)

Runs the editing inference. Connect the pipe output from the Loader and at least one image.

| Parameter | Description |
|-----------|-------------|
| `firered_fast_pipe` | Connect from the Loader output |
| `prompt` | Plain-language editing instruction (see examples below) |
| `seed` | Random seed for reproducibility |
| `steps` | Number of denoising steps. **Minimum 4** — 1 step produces a black image. 4 is the intended fast mode; 8–12 gives higher quality at the cost of speed. |
| `guidance_scale` | `true_cfg_scale`. Default `1.0`. Higher values (e.g. `3.0`–`5.0`) follow the prompt more strictly but may introduce artifacts. |
| `image1` | Primary image to edit (referred to as "Picture 1" in prompts) |
| `image2` | Optional second reference image ("Picture 2") — useful for style or clothing transfer |
| `width` / `height` | Output size in pixels. `0` = auto-computed from `image1`'s aspect ratio, targeting ~768×768 total pixels. Must be multiples of 8. |

---

### FireRed Fast Unloader

Explicitly frees the pipeline from CPU RAM and clears the session cache. Use this when you are done editing and want to load other large models.

---

## Example prompts

Single image editing:
```
Change the style to oil painting.
Make it look like a watercolor illustration.
Turn this into a pencil sketch.
Change the hair color to blonde.
Add sunglasses.
Make the background a sunset beach.
Remove the person from the background.
Make the image look like it was taken at night.
```

Two-image reference (connect both `image1` and `image2`):
```
Apply the style of Picture 2 to Picture 1.
Transfer the clothing from Picture 2 onto the person in Picture 1.
Make the person in Picture 1 wear the glasses from Picture 2.
```

---

## Recommended workflow

1. **Load Image** → connect to `image1` on the Sampler
2. **FireRed Fast Loader** → connect output to `firered_fast_pipe` on the Sampler
3. **FireRed Fast Sampler** → set `steps` to `4`, `guidance_scale` to `1.0`, write your prompt
4. Connect `IMAGE` output to **Save Image** or **Preview Image**
5. *(Optional)* Add **FireRed Fast Unloader** at the end of the workflow to free memory when done

Output resolution auto-scales from the input image's aspect ratio to ~768×768 pixels. Override with explicit `width`/`height` values if needed.

---

## Troubleshooting

**Black output image**
The most common cause is `steps = 1`. Set steps to at least **4**.

**Out of memory / process killed during load**
The transformer needs ~40 GB RAM to load. Close other applications, make sure no other large ComfyUI models are loaded, and try again. The node calls `unload_all_models()` automatically but other OS processes can compete for RAM.

**`guidance_scale` field shows the negative prompt text**
Your saved workflow has a stale value from an older version of this node. Clear the field and type `1.0`.

**`offload` dropdown still visible in the Loader**
Right-click the Loader node → Update node. The offload dropdown was removed in a refactor; `sequential_cpu_offload` is now the only strategy.

**Slow inference**
With `sequential_cpu_offload`, each denoising step moves hundreds of layers individually through the CPU→GPU→CPU cycle. At 4 steps on a 12 GB GPU expect **30–90 seconds** per generation. This is normal for a model this size on consumer hardware.

---

## Technical notes

The bundled `qwenimage/` directory contains a local copy of the pipeline and transformer classes adapted from the [diffusers](https://github.com/huggingface/diffusers) library. This is necessary because the upstream diffusers version may not yet have the exact pipeline variant used by FireRed-Image-Edit-1.1.

Loading uses `low_cpu_mem_usage=True` throughout, which enables safetensors memory-mapping for components that do not require dtype conversion. The text encoder in particular stays largely on disk until its pages are accessed during the first inference forward pass.
