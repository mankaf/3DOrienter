# TripoSR isolated worker setup

TripoSR is deliberately kept out of the 3DOrienter application environment. Its upstream
dependencies are older and include native/CUDA packages. The service invokes the official CLI
through `TripoSRBackend` with `shell=False` and a private job directory.

Pinned revisions:

- TripoSR: `107cefdc244c39106fa830359024f6a2f1c78871`
- torchmcubes: `879926d0ef58e6ce0ac2630fdecb5e53af7ed3ff`

## Checkout

```bash
git clone https://github.com/VAST-AI-Research/TripoSR.git /mnt/d/models/TripoSR
git -C /mnt/d/models/TripoSR checkout 107cefdc244c39106fa830359024f6a2f1c78871
uv venv --python 3.10 /mnt/d/models/TripoSR/.venv
```

The verified host configuration uses WSL2 because the Windows host does not currently have the
C++ and CUDA build tools required by torchmcubes. Neural inference uses CUDA; the pinned
torchmcubes wheel uses its CPU fallback for final marching cubes. Run
`workers/gpu/setup_triposr_wsl.sh` inside WSL to reproduce the worker environment. Do not install
TripoSR dependencies into the 3DOrienter application environment.

## Run through the offline pipeline

```bash
HF_HUB_DOWNLOAD_TIMEOUT=120 3dorienter-pipeline input.png \
  --backend triposr \
  --triposr-repo /mnt/d/models/TripoSR \
  --triposr-python /mnt/d/models/TripoSR/.venv/bin/python \
  --triposr-model /mnt/d/models/TripoSR-model \
  --jobs-dir /mnt/d/3dorienter-data/jobs \
  --target-max-mm 100
```

The TripoSR CLI remains private. Only the 3DOrienter worker may invoke it. Model downloads and
Hugging Face caches must live outside Git and be covered by the host retention and disk alerts.
The worker requires exclusive access to roughly 6 GB of VRAM; reject or defer jobs when another
application has reduced free VRAM below the configured safety threshold.
