# 3DOrienter CLI

A Python MVP for turning an image or existing mesh into a validated, scaled, and oriented FDM
artifact. It currently optimizes transparent geometric proxies for:

- support burden and removal effort,
- global sloped-surface quality,
- print height,
- first-layer contact/stability.

The project also emits labeled orientation datasets for a later AI model.

## Why simulation first, AI second

An AI model needs reliable target labels. This CLI is the label generator and baseline optimizer.
Once its scores correlate with slicer output and real prints, a neural model can learn to rank
orientations much faster and later incorporate material, printer, visual importance, and strength.

## Install

```bash
cd 3DOrienter
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

## Development checks

```bash
python -m ruff check .
python -m pytest -q
3dorienter --help
```

The GitHub Actions matrix runs these checks on Linux and Windows with Python 3.10 and 3.12.
To exercise the production-style CLI container locally:

```bash
docker build -t 3dorienter-cli .
docker run --rm 3dorienter-cli --help
```

## Offline image-to-STL pipeline

The deterministic mock backend exercises the complete pipeline without downloading model weights:

```bash
3dorienter-pipeline input.png \
  --backend mock \
  --jobs-dir ./data/jobs \
  --target-max-mm 100 \
  --samples 400
```

Each job gets a controlled private directory containing a metadata-stripped RGB input, generated
mesh, normalized STL, oriented STL, and versioned JSON report. The report uses relative artifact
names and does not expose host paths. See `docs/TRIPOSR_SETUP.md` for the isolated GPU backend.

## Optimize a model

```bash
3dorienter optimize part.stl \
  --output part_oriented.stl \
  --report part_report.json \
  --samples 600 \
  --max-overhang 45 \
  --layer-height 0.2
```

Lower scores are better. The output mesh is rotated, centered in XY, and translated onto Z=0.

## Analyze one rotation

```bash
3dorienter analyze part.stl --rotation 90 0 0
```

## Generate training labels

```bash
3dorienter make-dataset ./models --output orientations.jsonl --samples-per-model 300
```

Each JSONL row contains the model path, rotation matrix, quaternion, candidate type, total score,
and all component metrics.

## Current scoring model

### Support proxy

Faces are treated as support-requiring when they point downward beyond the configured overhang
threshold. The support score combines weighted overhang area with a projected-area × height-to-bed
volume proxy. This is fast but does not yet perform ray casting or model-aware support occlusion.

### Quality proxy

The quality term penalizes intermediate face slopes, where layer stair-stepping is most apparent.
Exactly vertical and exactly horizontal planes receive low slope cost. Downward supported surfaces
are additionally penalized through the support term.

### Stability and height

Large coplanar first-layer faces are rewarded, and tall orientations receive a small penalty.

## Recommended next milestones

1. Add ray/voxel-based support volume and accessibility/removal scoring.
2. Compare scores against PrusaSlicer or Cura CLI support volume and print time.
3. Add per-face importance masks so cosmetic surfaces matter more than hidden surfaces.
4. Generate a varied mesh corpus and split by model, never by orientation row.
5. Train a rotation-ranking model; start with mesh descriptors + gradient-boosted trees, then
   evaluate PointNet/DGCNN-style encoders only if the simpler baseline is insufficient.
6. Add anisotropic strength after defining loads, fixtures, material, infill, walls, and layer bond.

## Limitations

This is not yet a full slicing or finite-element simulation. It is an efficient orientation search
baseline designed to produce inspectable metrics and future ML labels.
