# FloodDiffusion Replication (Notebook Demo)

Replication-focused, runnable demo of **FloodDiffusion** for text-to-motion generation, centered on a single Colab/Jupyter notebook:

- [`FloodDiffusion_Colab_Demo.ipynb`](./FloodDiffusion_Colab_Demo.ipynb)
- `FloodDiffusion_Colab_Demo.ipynb - Colab.pdf` (static export)

This notebook demonstrates:
- loading the pretrained **`ShandaAI/FloodDiffusionTiny`** model from Hugging Face,
- generating 3D human motion from text,
- multi-prompt streaming transitions,
- visualizing the triangular diffusion-forcing schedule,
- rendering interactive 3D and MP4 motion outputs.

## What Is FloodDiffusion?

FloodDiffusion is a two-stage text-to-motion system:

1. **Motion VAE** compresses HumanML3D motion features (`263D`) into compact latent tokens (`4D`, temporal downsampling by `4x`).
2. **Latent Diffusion Forcing (LDF)** generates those latent tokens with a **triangular noise schedule** that supports progressive, streaming-style generation.

Paper: https://arxiv.org/abs/2512.03520  
Project page: https://shandaai.github.io/FloodDiffusion/

## Reported Benchmark Snapshot (HumanML3D, from paper)

| Method | FID ↓ | R-Precision ↑ | Diversity | MModality ↑ |
|---|---:|---:|---:|---:|
| MDM (2023) | 0.544 | 0.611 | 9.559 | 2.799 |
| MotionDiffuse (2023) | 0.630 | 0.782 | 9.410 | 1.553 |
| MLD (2023) | 0.473 | 0.772 | 9.724 | 2.413 |
| ReMoDiffuse (2023) | 0.103 | 0.795 | 9.018 | 1.795 |
| MotionLCM (2024) | 0.424 | 0.786 | 9.576 | 2.058 |
| **FloodDiffusion** | **0.089** | **0.797** | **9.424** | **2.463** |

These values are paper-reported metrics and are reproduced in the notebook plots for reference.

## Repository Scope

This repository is currently **not** a full training codebase.  
It is a **self-contained educational + inference notebook** with conceptual training walkthroughs.

Training commands referenced in the notebook (for example `pretokenize_vae.py`) are explained for context but are not present as scripts in this repo.

## Security Note

The notebook loads the pretrained model with `trust_remote_code=True`.  
Review model repository code before using this setting in production or shared environments.

## Quick Start (Colab)

1. Open `FloodDiffusion_Colab_Demo.ipynb` in Google Colab.
2. Optional: set runtime to GPU (`T4` or better) for faster inference.
3. Run cells top-to-bottom.
4. First run downloads model weights from Hugging Face (about 1-2 GB for Tiny).

## Quick Start (Local Jupyter)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers huggingface_hub sentencepiece protobuf
pip install imageio imageio-ffmpeg matplotlib numpy plotly ipywidgets jupyter
jupyter lab
```

Then open and run:

```text
FloodDiffusion_Colab_Demo.ipynb
```

## Core Inference API (Used in Notebook)

Single prompt:

```python
joints = model(
    "a person walking forward",
    length=30,                 # latent tokens
    output_joints=True,        # returns (frames, 22, 3)
    smoothing_alpha=0.5
)
```

Multi-text transition:

```python
multi_joints = model(
    text=["walk forward", "turn around", "raise both hands"],
    text_end=[20, 40, 60],     # latent token boundaries
    length=60,
    output_joints=True,
    smoothing_alpha=0.5
)
```

Output format after normalization:
- shape: `(frames, 22, 3)`
- coordinates: 22-joint HumanML3D skeleton

## Notebook Walkthrough

1. **Setup & model loading**  
   Installs dependencies, downloads model checkpoint, applies compatibility patches, loads remote code.
2. **Architecture deep dive**  
   VAE + LDF components, parameter inspection, triangular schedule intuition.
3. **Inference**  
   Single prompt generation, multi-prompt transitions, comparative prompt tests.
4. **Training overview (conceptual)**  
   Three-phase explanation (VAE, pre-tokenization, LDF) with illustrative pseudo-code.
5. **Results discussion**  
   Paper-reported HumanML3D metrics and qualitative analysis plots.
6. **Visualization demo**  
   Plotly 3D viewer, matplotlib animation, MP4 rendering.

## Compatibility Notes

The notebook patches cached model files to improve portability in CPU-only / non-flash-attention environments:
- CUDA device fallback in `t5.py`,
- flash-attention fallback in `wan_model.py`,
- dtype-safe return in `attention.py`,
- `AutoModel` to `UMT5EncoderModel` adjustment in `diffusion_forcing_wan_tiny.py`.

These patches are applied automatically during the model-loading cell.

## Troubleshooting

- **Long first run**: expected due to checkpoint download and model initialization.
- **CUDA autocast warnings on CPU**: expected; inference still runs.
- **Slow generation**: use GPU runtime in Colab, reduce `length`, or increase `smoothing_alpha` only if needed.
- **Video codec warnings (`pix_fmt`)**: non-fatal; MP4 output is still produced in notebook runs.

## Reproducibility Tips

- Keep package versions stable in your environment.
- Run cells sequentially from top to bottom.
- Re-run the model-loading cell if your runtime resets.
- For consistent comparisons, keep prompt text, `length`, and `smoothing_alpha` fixed.

## Citation

If you use this notebook/demo, please cite the FloodDiffusion paper:

```bibtex
@article{cai2025flooddiffusion,
  title   = {FloodDiffusion: Tailored Diffusion Forcing for Streaming Motion Generation},
  author  = {Cai, Yiyi and Wu, Yuhan and Li, Kunhang and Zhou, You and Zheng, Bo and Liu, Haiyang},
  journal = {arXiv preprint arXiv:2512.03520},
  year    = {2025}
}
```

## Acknowledgements

- FloodDiffusion authors and project maintainers
- HumanML3D benchmark creators
- Diffusion Forcing and WAN architecture contributors
