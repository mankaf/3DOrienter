#!/usr/bin/env bash
set -euo pipefail

triposr_revision="107cefdc244c39106fa830359024f6a2f1c78871"
torchmcubes_revision="879926d0ef58e6ce0ac2630fdecb5e53af7ed3ff"
triposr_root="${TRIPOSR_ROOT:-/mnt/d/models/TripoSR}"
uv_bin="${UV_BIN:-$HOME/.local/bin/uv}"
worker_python="$triposr_root/.venv/bin/python"

if [[ ! -x "$uv_bin" ]]; then
  echo "uv is required; set UV_BIN to its absolute path" >&2
  exit 2
fi

if [[ ! -e "$triposr_root" ]]; then
  mkdir -p "$(dirname "$triposr_root")"
  git clone https://github.com/VAST-AI-Research/TripoSR.git "$triposr_root"
elif [[ ! -d "$triposr_root/.git" ]]; then
  echo "TRIPOSR_ROOT exists but is not a Git checkout: $triposr_root" >&2
  exit 2
fi

if [[ -n "$(git -C "$triposr_root" status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing to change a dirty TripoSR checkout" >&2
  exit 2
fi
git -C "$triposr_root" fetch origin "$triposr_revision"
git -C "$triposr_root" checkout --detach "$triposr_revision"

if [[ ! -x "$worker_python" ]]; then
  "$uv_bin" venv --python 3.10 "$triposr_root/.venv"
fi

"$uv_bin" pip install --python "$worker_python" \
  torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

if ! "$worker_python" -c "import torchmcubes" >/dev/null 2>&1; then
  build_root="$(mktemp -d -t torchmcubes-build.XXXXXXXX)"
  cleanup() {
    rm -rf -- "$build_root"
  }
  trap cleanup EXIT

  git clone https://github.com/tatsy/torchmcubes.git "$build_root/src"
  git -C "$build_root/src" checkout --detach "$torchmcubes_revision"
  "$uv_bin" venv --python 3.10 "$build_root/venv"
  "$uv_bin" pip install --python "$build_root/venv/bin/python" \
    torch==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
  "$uv_bin" pip install --python "$build_root/venv/bin/python" \
    build==1.6.1 cmake==4.4.3 ninja==1.13.2 pybind11==3.1.0 scikit-build-core==1.0.3
  (
    cd "$build_root/src"
    "$build_root/venv/bin/python" -m build --wheel --no-isolation \
      --outdir "$build_root/dist"
  )
  "$uv_bin" pip install --python "$worker_python" --no-deps "$build_root"/dist/*.whl
fi

"$uv_bin" pip install --python "$worker_python" \
  omegaconf==2.3.0 Pillow==10.1.0 einops==0.7.0 transformers==4.35.0 \
  trimesh==4.0.5 rembg==2.0.69 huggingface-hub==0.17.3 imageio==2.37.4 \
  imageio-ffmpeg==0.6.0 gradio==4.8.0 xatlas==0.0.9 moderngl==5.10.0 \
  onnxruntime==1.23.2

"$worker_python" -c \
  "import torch, torchmcubes; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print('torchmcubes_cuda=', torchmcubes.HAS_CUDA)"
(
  cd "$triposr_root"
  "$worker_python" run.py --help >/dev/null
)

echo "TripoSR worker is ready at $triposr_root"
