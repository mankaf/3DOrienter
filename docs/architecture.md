# Architecture

## Current pipeline

1. Load and conservatively repair a triangle mesh.
2. Precompute face normals, areas, centroids, vertices, and scale references.
3. Generate candidate rotations:
   - identity,
   - the 24 axis-aligned cube rotations,
   - rotations placing large faces on the build plate,
   - uniformly random rotations in SO(3).
4. Score every candidate using support, quality, height, and stability proxies.
5. Export the best rotated mesh and an inspectable JSON report.
6. Optionally write every candidate and score to JSONL as ML supervision.

## ML plan

The learning task should be formulated as candidate ranking rather than direct Euler-angle
regression. Rotations are periodic and non-unique; ranking candidate quaternions avoids many
representation problems.

A practical sequence is:

1. Build a dataset split by model identity.
2. Train a non-neural baseline on mesh descriptors plus candidate-relative normal histograms.
3. Measure top-k regret: score(predicted best) - score(simulated best).
4. Add a point-cloud or mesh encoder only after the baseline is established.
5. Use the learned ranker to propose a small candidate set, then verify it with the simulator.

This hybrid approach keeps the final decision explainable and prevents the model from silently
selecting an orientation outside the simulation's validated behavior.

## Planned support-removal simulation

The present support-volume proxy treats overhang facets independently. The next version should:

- project overhang samples toward the build plate,
- ray-test whether the model itself provides support,
- voxelize or rasterize support columns,
- estimate interface area and separate support islands,
- estimate tool accessibility for breaking or dissolving supports.

## Planned strength simulation

Strength cannot be inferred from geometry alone. It needs a load case and print process inputs:
material, layer adhesion, perimeter count, infill pattern/density, nozzle width, temperatures,
fixtures, and applied forces. A first strength metric can use anisotropic material tensors in a
coarse voxel finite-element model; later versions can calibrate those tensors from printed coupons.
