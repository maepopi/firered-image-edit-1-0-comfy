# FireRed Image Edit 1.0 — ComfyUI Custom Node

A ComfyUI custom node wrapping the **FireRed-Image-Edit-1.1** pipeline with the fast **Qwen-Image-Edit-Rapid-AIO-V19** transformer, enabling high-quality image editing in as few as **4 inference steps**.

Based on the [FireRed-Image-Edit-1.0-Fast](https://huggingface.co/spaces/prithivMLmods/FireRed-Image-Edit-1.0-Fast) HuggingFace Space by prithivMLmods.

---

## Models used

| Role | Model |
|---|---|
| Base pipeline (VAE, scheduler, text encoder) | [FireRedTeam/FireRed-Image-Edit-1.1](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1) |
| Fast transformer | [prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19](https://huggingface.co/prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19) |

Both are downloaded automatically on first use and cached in your ComfyUI `models/diffusers/` folder.

---

## Installation

### Via ComfyUI Manager
Search for `FireRed Image Edit` and install.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/maepopi/firered-image-edit-1-0
cd firered-image-edit-1-0
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` pulls the latest `diffusers` and `accelerate` from GitHub, which is required for the QwenImage pipeline classes.

---

## Nodes

All nodes appear under the **`FireRedEdit/Fast`** category.

### FireRed Fast Loader (1.1)

Downloads and loads the pipeline. The pipeline is cached between runs — if settings are unchanged, subsequent runs skip reloading entirely.

| Input | Type | Default | Description |
|---|---|---|---|
| `base_model_id` | STRING | `FireRedTeam/FireRed-Image-Edit-1.1` | HuggingFace repo ID or local path for the base pipeline |
| `fast_transformer_id` | STRING | `prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19` | HuggingFace repo ID or local path for the fast transformer |
| `precision` | bf16 / fp16 | `bf16` | Weight dtype — bf16 recommended |
| `offload` | dropdown | `model_cpu_offload` | Memory strategy (see below) |
| `enable_fa3` | BOOLEAN | `False` | Enable Flash Attention 3 (requires `flash-attn` installed) |

**Offload options:**
- `model_cpu_offload` — moves model components to CPU when not in use (good balance of speed and VRAM)
- `sequential_cpu_offload` — more aggressive offloading, slower but lower peak VRAM
- `full_gpu` — keep everything on GPU, fastest but requires most VRAM

**Output:** `FIRERED_FAST_PIPE` — pass to the sampler node.

---

### FireRed Fast Sampler (1.1)

Runs the editing inference. Connect one or two images and describe your edit in the prompt.

| Input | Type | Default | Description |
|---|---|---|---|
| `firered_fast_pipe` | FIRERED_FAST_PIPE | required | Pipeline from the Loader node |
| `prompt` | STRING | — | Editing instruction |
| `seed` | INT | 0 | Reproducibility seed |
| `steps` | INT | 4 | Inference steps — 4 is the intended fast mode |
| `guidance_scale` | FLOAT | 1.0 | `true_cfg_scale` — 1.0 works best for the distilled model |
| `image1` | IMAGE | optional | Primary image to edit (referenced as "Picture 1") |
| `image2` | IMAGE | optional | Secondary reference image (referenced as "Picture 2") |
| `negative_prompt` | STRING | *see node* | Negative conditioning |
| `width` | INT | 0 | Output width — 0 = auto from image1 aspect ratio |
| `height` | INT | 0 | Output height — 0 = auto from image1 aspect ratio |

**Output:** `IMAGE`

#### Multi-image editing
When two images are connected, the model sees them as **Picture 1** and **Picture 2**. You can reference them directly in the prompt:

```
Transfer the glasses from Picture 2 onto the person in Picture 1, keep everything else the same.
```

```
Dress the person in Picture 1 with the outfit from Picture 2, preserve the face and pose.
```

#### Single-image editing
```
Convert to black and white with high contrast.
Convert to a dotted cartoon style.
Apply a cinematic polaroid look with warm tones.
Make the background a snowy mountain landscape.
```

---

### FireRed Fast Unloader (1.1)

Explicitly frees the pipeline from GPU and CPU memory. Useful when switching to other large models.

| Input | Type | Description |
|---|---|---|
| `firered_fast_pipe` | FIRERED_FAST_PIPE | Pipeline to unload |

---

## Typical workflow

```
[Load Image] ──────────────────────────────────────────┐
                                                        ▼
[FireRed Fast Loader (1.1)] ──► [FireRed Fast Sampler (1.1)] ──► [Save Image]
                                        ▲
                             prompt: "Make it look like an oil painting"
                             steps: 4
                             guidance_scale: 1.0
```

---

## Tips

- **4 steps is enough.** The fast transformer is distilled for low-step inference — pushing steps beyond 10–15 rarely improves quality meaningfully.
- **Keep `guidance_scale` at 1.0.** Higher values can over-saturate results with this distilled model.
- **Output resolution** is auto-calculated to ~1024×1024 total pixels while preserving the input aspect ratio, matching the original Space behavior.
- **Flash Attention 3** (`enable_fa3=True`) can speed up inference if you have `flash-attn` installed and a compatible GPU (Ampere/Ada/Hopper).

---

## Requirements

- Python 3.10+
- PyTorch with CUDA
- See `requirements.txt` for Python dependencies

---

## Credits

- Pipeline: [FireRedTeam](https://huggingface.co/FireRedTeam)
- Fast transformer & Space: [prithivMLmods](https://huggingface.co/prithivMLmods)
- Bundled `qwenimage` module from the [FireRed-Image-Edit-1.0-Fast Space](https://huggingface.co/spaces/prithivMLmods/FireRed-Image-Edit-1.0-Fast)
