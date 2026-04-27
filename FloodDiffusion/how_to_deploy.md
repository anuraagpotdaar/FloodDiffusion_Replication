# How to deploy this Space to Hugging Face

This repo IS a Hugging Face Space — every push to its `main` branch on
`huggingface.co` triggers a Docker build and redeploy. Same workflow as a
normal git remote.

## One-time setup

1. **Create the Space on huggingface.co**
   - Go to <https://huggingface.co/new-space>
   - Name it (e.g. `FloodDiffusion-Motion`)
   - SDK: **Docker**
   - Hardware: **T4 small** (CUDA Dockerfile expects an NVIDIA GPU; CPU works
     too but is much slower — see "CPU-only" below if you change tier).
   - Visibility: public or private — your call.

2. **Get an HF access token** (write scope) at
   <https://huggingface.co/settings/tokens>. Cache it locally:
   ```bash
   huggingface-cli login         # interactive, stores in ~/.cache/huggingface/token
   # or just write the token directly:
   echo "$YOUR_TOKEN" > ~/.cache/huggingface/token
   ```

3. **Install Git LFS** (the rigged mesh `static/models/Soldier.glb` is tracked
   via LFS — Hugging Face requires it for binaries):
   ```bash
   brew install git-lfs   # macOS, or apt-get install git-lfs on Linux
   git lfs install        # one-time per machine
   ```

## First push

```bash
cd FloodDiffusion

# Point the repo at YOUR Space (replace USERNAME and SPACE_NAME)
HF_TOKEN_VAL=$(cat ~/.cache/huggingface/token)
git remote add hf "https://USERNAME:${HF_TOKEN_VAL}@huggingface.co/spaces/USERNAME/SPACE_NAME"

# (Optional) sanity-check what's about to ship
git status
git lfs ls-files

# Push — the very first push usually needs --force because HF creates the
# Space with an initial commit that's unrelated to our local history.
git push -u hf main:main --force
```

After the push:
- The Space rebuilds automatically (3–10 min for the CUDA image).
- Build progress is at `https://huggingface.co/spaces/USERNAME/SPACE_NAME`
  → "Logs" tab.
- First request after build pulls `ShandaAI/FloodDiffusionTiny` (~500 MB) into
  the container's HF cache. Subsequent loads are instant.

## Subsequent updates

Just commit and push — no `--force` needed once histories agree:

```bash
git add path/to/changed/files
git commit -m "your change"
git push hf main:main
```

The Space picks up the push, rebuilds the Docker image, and redeploys.

## Strip token from remote URL after the first push (optional but recommended)

```bash
git remote set-url hf https://huggingface.co/spaces/USERNAME/SPACE_NAME
```

Subsequent `git push` will then prompt for credentials (or use the credential
helper / `HF_TOKEN` env var with `huggingface_hub`).

## Useful checks before pushing

| Check | Command | Why |
| --- | --- | --- |
| Frontmatter is valid | `head -15 README.md` | HF rejects pushes with bad YAML; `short_description` ≤ 60 chars, `app_port`/`sdk`/`hardware` must be set. |
| Binaries are LFS-tracked | `git lfs ls-files` | HF rejects raw binaries > ~1 MB. |
| Dockerfile exposes the right port | `grep EXPOSE Dockerfile` | Must match `app_port:` in README frontmatter (here: 7860). |
| App boots with `python app.py --port 7860` | `CMD` line in `Dockerfile` | Same port as `app_port`. |

## CPU-only deployment

If you change the Space hardware to a CPU tier, the app still works (the
device-selection logic in `model_manager.py` falls back to CPU when CUDA isn't
detected) but generation is several seconds per chunk. Optionally swap the
Dockerfile base image to `python:3.10-slim` and remove the CUDA torch wheels:

```dockerfile
FROM python:3.10-slim
# … remove the `torch* --index-url cu124` line, keep the rest of pip install
```

## Common errors

- **`pre-receive hook declined: Your push was rejected because it contains binary files`**
  → A `.glb`, `.safetensors`, etc. wasn't tracked by LFS. Add the extension to
  `.gitattributes` (`*.glb filter=lfs diff=lfs merge=lfs -text`), `git rm
  --cached <file>`, `git add <file>`, `git commit --amend --no-edit`, push
  again.

- **`pre-receive hook declined: short_description length must be less than or equal to 60 characters`**
  → Shorten `short_description:` in the README frontmatter, amend, push.

- **`! [rejected] main -> main (fetch first)`**
  → Remote has different history (HF's initial template commit). Use
  `--force` on the first push only.

- **Space build fails with `nvidia-smi: command not found` or CUDA OOM**
  → You're on a CPU tier with the CUDA Dockerfile. Either change hardware to
  T4 or swap to a CPU base image.

- **Space starts but the `/` route 404s**
  → Templates/static aren't being copied. Confirm `COPY --chown=user:user . .`
  is in the Dockerfile and you're not gitignoring `templates/` or `static/`.

## What actually gets deployed

Everything in the working tree at the time of `git push hf main`. The
gitignore in this repo deliberately excludes runtime artefacts (`app.log`,
`server.pid`, `telemetry.jsonl`, `__pycache__/`, `.venv/`) so they don't ship.
The shipped image contains:

- `app.py`, `model_manager.py`, `motion_process.py`, `telemetry.py`
- `static/` (CSS, JS including `mesh_body.js`, `Soldier.glb` via LFS)
- `templates/index.html`
- `Dockerfile`, `requirements.txt`, `README.md`, `.gitattributes`, `.gitignore`
- `test_*.sh` (CLI tests; harmless if they ride along)
