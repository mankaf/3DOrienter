# 3DOrienter hosted service deployment plan

**Status:** proposed architecture and gated implementation plan  
**Prepared:** 2026-09-18  
**Target repository:** `D:\github\3DOrienter`

## 1. Executive decision

Build a web service that turns one or more reference images into a printable mesh, validates it, lets the user choose an orientation objective, and returns an oriented STL plus an explainable report.

Do **not** make Hunyuan3D 2.1 the production dependency from Prague without separate written permission. Tencent's current license says it does not apply in the EU, UK, or South Korea and prohibits use and output outside its defined territory. It explicitly includes hosted services. This is a launch blocker, not a minor attribution issue.

Use a model adapter with this initial order:

1. **Recommended benchmark winner candidate:** Stable Fast 3D. It documents about 6 GB VRAM and is likely the stronger product candidate, but its Stability AI Community License requires commercial registration, attribution, and an enterprise license above the stated USD 1 million annual-revenue threshold.
2. **MVP/fallback:** TripoSR as the simplest legal/technical baseline. Its project states that code and pretrained model are MIT licensed and that a default run uses about 6 GB VRAM.
3. **Hunyuan3D 2.1:** disabled by default; enable only when counsel confirms written rights covering the operator, EU location, users, hosted service, and generated outputs.
4. **External API fallback:** optional provider adapter for overflow or higher-quality tiers, with explicit per-job cost and data-processing terms.

The RTX 4070 should be a private, single-concurrency GPU worker. Host the public control plane separately. Never expose ComfyUI directly to the internet.

### Model shortlist for the RTX 4070

| Model | Published VRAM | Commercial/license posture | 4070 verdict | Role |
|---|---:|---|---|---|
| Stable Fast 3D | About 6 GB | Community license permits hosted commercial use below its revenue threshold after registration; attribution/notice obligations apply | Good fit; benchmark first | Preferred quality candidate |
| TripoSR | About 6 GB | Project states code and pretrained model are MIT licensed | Good fit and simplest compliance | Baseline/fallback |
| Hunyuan3D 2.1 Shape | About 10 GB | Community license territory excludes EU/UK/South Korea | Technical fit is tight; legal blocker in Prague | Disabled absent separate rights |
| Hunyuan3D 2.1 Texture | About 21 GB | Same territory blocker | Does not fit 12 GB normally | Exclude |
| Microsoft TRELLIS | At least 16 GB | Models and most code MIT, with submodule exceptions | Does not fit the 12 GB card as documented | Future cloud-GPU benchmark |

**Recommendation:** package Stable Fast 3D and TripoSR behind the same adapter, run the fixed 20-image corpus blind, and choose on printable geometry—not beauty renders. Measure watertightness, disconnected components, minimum feature survival, dimensional consistency after scaling, orientation/slicer success, p95 runtime, and human preference. If Stable Fast 3D's gain is small, ship TripoSR because MIT licensing and fewer obligations reduce operational risk. If its geometry is materially better, register for the Stability AI Community License, implement the required notices/attribution, and make it primary.

## 2. Product definition

### Core promise

“Upload a product/object photo, receive a generated printable STL, compare practical print orientations, and download the selected result with a transparent report.”

Avoid claims such as “structurally safe,” “engineering validated,” or “minimum supports” until the scoring system is calibrated against real slicer output and physical tests.

### User journey

1. User creates a project and accepts content/IP terms.
2. User uploads one primary image and optionally extra views.
3. Preflight checks file type, dimensions, transparency/background, moderation, and likely subject quality.
4. User supplies a physical target size because a single image has no reliable real-world scale.
5. A paid generation attempt is queued.
6. The service generates a mesh and produces turntable renders plus mesh diagnostics.
7. User can:
   - accept the mesh;
   - upload a better/different picture and create another generation attempt;
   - reject a technically failed attempt under the retry policy.
8. User configures orientation using a simple preset or advanced controls.
9. The service generates several orientation candidates, validates the finalists with a real slicer, and displays trade-offs.
10. User downloads the oriented STL, original generated mesh, preview renders, and JSON/PDF report.

### Retry semantics

Separate expensive **generation attempts** from cheap **orientation runs**:

- One generation credit = one accepted GPU reconstruction request.
- Orientation reruns on the same mesh are included and rate-limited.
- Re-uploading a different image creates a new version and normally consumes one generation credit.
- Give one automatic replacement credit when the service quality gate detects a technical failure: empty/corrupt mesh, non-manifold beyond repair threshold, severe fragmentation, timeout, or worker crash.
- A subjective dislike is not an unlimited free retry. Offer bundles and a clearly priced “try another photo” action.
- Preserve version history so users can return to an earlier result.

## 3. Orientation experience

### Simple mode

Offer four presets with plain-language trade-offs:

| Preset | Primary goal | Required input | User-facing caveat |
|---|---|---|---|
| Easy support removal | Accessibility and low interface burden | Printer/support profile | Estimate until slicer validation is complete |
| Surface quality | Protect visible faces from supports/stair-stepping | User marks important side(s) | Cosmetic importance is user-supplied |
| Fast/economical | Low print time and material | Printer/material profile | Estimate varies by slicer and printer |
| Strength-aware | Favor layer direction for a stated load | Load arrow, force category, material/profile | Guidance only; not structural certification |

### Advanced mode

- Printer build volume and nozzle/layer profile.
- Material, walls, infill percentage/pattern.
- Normal/tree supports and overhang threshold.
- Load direction in a 3D viewer; optional “unknown” state.
- Visible/cosmetic face selection.
- Weighted sliders with a total normalized to 100%.
- Candidate count and search depth hidden behind an expert toggle.

### Result comparison

Show 3–5 candidates, not only one opaque answer. Each card should include:

- interactive orientation preview;
- dimensions and bed-contact area;
- slicer-estimated print time, filament, and support material;
- support-contact/accessibility estimate;
- quality and strength-aware heuristic scores;
- concise reason for recommendation;
- “estimated” badges and confidence/validation state.

The current `ai-optimize` command uses geometric proxies and an LM Studio ranker. In production, deterministic scoring plus slicer measurements should select the result. An LLM may explain trade-offs but should not be the authoritative numerical decision maker.

## 4. System architecture

See `service-architecture.html` in the same directory for the standalone architecture diagram.

### Public control plane

- **Web:** Next.js/TypeScript, hosted on a managed platform or small EU VPS behind a CDN/WAF.
- **API:** FastAPI/Python with OpenAPI, stateless containers, strict request validation.
- **Database:** managed PostgreSQL for users, projects, versions, jobs, credits, and audit events.
- **Queue:** Redis-backed durable job queue for MVP; migrate to a managed queue when needed.
- **Object storage:** EU-region S3-compatible private buckets; all access via short-lived signed URLs.
- **Billing:** Stripe Checkout/customer portal/webhooks; keep card data out of the application.
- **Email:** transactional provider for verification, receipts, and job completion.

### Private compute plane

- A worker agent on the 4070 machine opens an outbound authenticated connection or polls the queue over TLS/VPN.
- The worker downloads a signed input object, runs one containerized workflow, uploads artifacts, records metrics, then erases the local job directory.
- Concurrency is **one GPU generation at a time** on a 12 GB 4070.
- CPU orientation/slicer work may overlap only after benchmarks prove it does not starve GPU preprocessing or disk I/O.
- ComfyUI, if retained, binds to localhost/private network only. Prefer a versioned workflow JSON or direct Python inference API over UI automation.
- A cloud GPU provider implements the same worker contract for maintenance and overflow.

### Pipeline

1. `preflight`: decode image safely, normalize EXIF orientation, remove metadata, create canonical PNG/WebP.
2. `reconstruct`: selected image-to-3D model produces GLB/OBJ/mesh.
3. `mesh_validate`: scene flattening, connected-component checks, degenerate-face removal, normals, watertightness diagnostics.
4. `mesh_scale`: apply user-supplied physical size and units.
5. `preview`: render turntable and thumbnails.
6. `orient`: generate candidates with 3DOrienter.
7. `slice_validate`: run PrusaSlicer or OrcaSlicer CLI against finalists and parse time/material/support metrics.
8. `package`: oriented STL, original mesh, previews, and report manifest.
9. `publish`: upload artifacts and transition job atomically to `ready`.

## 5. Job state model

Use explicit idempotent states:

`created -> upload_pending -> preflight -> queued -> generating -> mesh_validation -> preview_ready -> orienting -> slicing -> ready`

Terminal/side states:

- `rejected`: policy or unsafe input.
- `failed_retryable`: infrastructure failure; no credit consumed or credit restored.
- `failed_quality_gate`: automatic replacement credit.
- `failed_user_input`: unsupported/insufficient input; no automatic claim of model failure.
- `cancelled`: allowed before GPU execution begins.
- `expired`: artifacts passed retention date.

Every stage records `attempt_id`, code/model/workflow versions, start/end time, worker ID, and structured error. Queue delivery may happen more than once; artifact keys and transitions must therefore be idempotent.

## 6. API surface

Initial versioned endpoints:

- `POST /v1/projects`
- `POST /v1/projects/{id}/sources/upload-url`
- `POST /v1/projects/{id}/generation-attempts`
- `GET /v1/jobs/{id}`
- `POST /v1/generations/{id}/orientation-runs`
- `GET /v1/generations/{id}/candidates`
- `POST /v1/orientation-runs/{id}/select`
- `POST /v1/artifacts/{id}/download-url`
- `POST /v1/billing/checkout`
- `POST /v1/webhooks/stripe`
- `DELETE /v1/projects/{id}`

Use UUID/ULID identifiers; never expose local paths. Enforce ownership on every lookup. Webhooks require signature verification and replay protection.

## 7. Core data model

- `users`: identity, region, terms version, deletion state.
- `projects`: owner, title, printer profile, retention policy.
- `source_assets`: immutable object key, hash, media metadata, moderation state.
- `generation_attempts`: source version, provider/model/workflow versions, seed/settings, credit transaction, status.
- `meshes`: generated artifact, units/scale, topology diagnostics, bounding box.
- `orientation_runs`: mesh, objective weights, load case, printer/slicer profile, 3DOrienter version.
- `orientation_candidates`: rotation matrix, metrics, slicer measurements, rank, selected flag.
- `artifacts`: type, object key, hash, byte size, expiry.
- `jobs` and `job_events`: durable state and append-only execution history.
- `credit_ledger`: immutable grants, purchases, reservations, consumption, refunds.
- `audit_events`: security and administrative events without user content.

## 8. Repository migration and target layout

Migration status on 2026-09-18: the allowlisted CLI source has been copied into the target Git repository while the original source directory remains unchanged. The source directory also contains roughly 12.3 GB of model files, 7.0 GB of training environment data, 1.2 GB of training runs, virtual environments, generated meshes, and a local `.env` file; all remain excluded from Git.

Proposed monorepo:

```text
3DOrienter/
  apps/
    web/                  # Next.js UI
    api/                  # FastAPI control plane
  packages/
    orienter3d/           # migrated Python package
    contracts/            # JSON schemas/OpenAPI-generated types
  workers/
    gpu/                   # reconstruction adapter + workflow runner
    mesh/                  # validation, orientation, slicing
  infra/
    compose/               # local integration stack
    terraform/             # hosted control plane later
  workflows/               # versioned ComfyUI JSON, if used
  tests/
    fixtures/              # tiny redistributable meshes/images only
    integration/
    e2e/
  docs/
  LICENSE
  README.md
```

Migration allowlist from `D:\Downloads 2026\3dorienter-cli`:

- `src/orienter3d/**` excluding `__pycache__` and generated `*.egg-info`.
- `tests/**`, `docs/**`, `examples/**`.
- `pyproject.toml`, `README.md`, `LICENSE`, and an expanded `.gitignore`.

Explicit exclusions:

- `.env*`, `.venv`, `.train-venv`, credentials, local URLs.
- `models`, `training_runs`, `training_data`, caches, weights.
- STL/3MF/user files, generated reports, previews, and output artifacts.
- Any model with unclear provenance or redistribution rights.

Store weights outside Git with a pinned manifest containing source, exact revision/hash, license, acceptance date, and download procedure. Use Git LFS only for small licensed fixtures—not multi-gigabyte model weights.

## 9. Changes required in 3DOrienter

1. Split reusable library functions from Typer CLI presentation.
2. Replace filesystem-path assumptions with byte streams or controlled job directories.
3. Add Pydantic request/result schemas and a stable machine-readable error taxonomy.
4. Remove `.env` parsing and LM Studio URL details from report output.
5. Make deterministic simulator ranking authoritative; keep AI explanation optional.
6. Add mesh complexity limits before expensive operations.
7. Add real slicer adapters and record slicer/profile versions.
8. Improve support accessibility with ray/voxel analysis.
9. Define strength-aware scoring as a heuristic tied to an explicit load case; calibrate before marketing it.
10. Add golden fixtures, malformed mesh tests, property tests for rotations, and report schema versioning.
11. Package Linux containers with pinned Python/CUDA dependencies; do not depend on a WSL-created virtual environment.
12. Emit progress events and honor cancellation between stages.

Current baseline verification on 2026-09-18: all three existing unit tests, Ruff, CLI startup, and an end-to-end STL optimization smoke test pass in a fresh Windows Python 3.14 environment. Cross-platform GitHub Actions covers Linux and Windows on Python 3.10 and 3.12. This is useful but still far below a service launch gate.

## 10. GPU feasibility and capacity

The RTX 4070 normally has 12 GB VRAM. Official Hunyuan3D 2.1 documentation states approximately 10 GB for shape generation, 21 GB for texture generation, and 29 GB for both. Even apart from the license blocker, only shape generation is nominally plausible, with little headroom and concurrency one. Texture generation does not fit normally.

TripoSR's documented default is about 6 GB VRAM, making it a better initial feasibility baseline. Benchmark on the actual machine; published A100 timings are not capacity estimates for a 4070.

For an STL product, omit the texture/PBR stage entirely: STL carries geometry, not PBR materials, and the official texture stage's approximately 21 GB requirement exceeds the card. On the actual worker, plan for at least **100?150 GB free disk** for environments, model cache, outputs, and logs. Treat **32 GB system RAM** as the practical shape-only floor; this RAM figure is an operational estimate rather than an official Tencent requirement.

Run the GPU headlessly where possible and close other CUDA/display-heavy applications. Hunyuan shape generation's published 10 GB requirement leaves little margin for CUDA workspaces and fragmentation on a 12 GB card. Regardless of model, keep production concurrency at one until measured peak VRAM and 25-job soak tests establish a safe limit.

Required benchmark matrix:

- 20 representative images: simple object, thin parts, reflective, transparent, organic, cluttered background, multiple views.
- Cold and warm starts.
- Peak VRAM/RAM, wall time by stage, GPU utilization, output size, failure type.
- 25 sequential jobs to find leaks and thermal throttling.
- Concurrent CPU slicing/orientation on and off.
- Recovery after process kill, CUDA OOM, and machine restart.

Capacity formula:

`jobs/hour = 3600 / p95_generation_seconds * target_utilization`

Use target utilization of 0.6–0.7 for an interactive service and maintenance headroom. Do not publish an SLA until p95 completion and uptime are measured for at least two weeks.

## 11. Monetization proposal

### Recommended charging unit

Charge for **generation credits**, not downloads or orientation changes. The GPU reconstruction is the scarce action; orientation on an existing mesh is relatively cheap and encourages users to engage with 3DOrienter's differentiator.

### Initial offers for validation

- **Free account:** one preview attempt after email verification; rendered preview only, aggressive rate limits.
- **Starter pack:** EUR 7 for 5 generation credits, valid 90 days.
- **Creator:** EUR 15/month for 15 credits, normal queue, private projects.
- **Pro:** EUR 39/month for 50 credits, priority queue and longer retention.
- **Overflow/high-quality option:** separately priced according to external GPU/API cost; never silently spend external-provider money.

These are hypotheses, not final prices. Validate willingness to pay with a landing page and 10–20 design partners before building subscription complexity. Packs are a simpler launch mechanism than recurring plans.

### Unit economics ledger

For every job record:

- GPU seconds and estimated workstation kWh.
- cloud/API spend if used.
- object storage GB-days and egress.
- payment fee allocation.
- automatic retry and support/refund cost.
- gross revenue net of VAT and payment fees.

`contribution margin = net revenue - variable compute - storage/egress - payment fees - retry reserve - support reserve`

Electricity is unlikely to be the dominant cost on a 4070; payment fees, failed attempts, support, acquisition, VAT/accounting, and owner time can dominate. Define a minimum checkout amount to avoid poor economics on one-credit payments.

## 12. Security, privacy, and abuse controls

- TLS everywhere; private buckets; least-privilege service accounts.
- Worker has no inbound public port and no broad cloud credentials.
- Signed object URLs expire quickly and are scoped to one object/action.
- Allow only JPEG/PNG/WebP initially; verify magic bytes after decoding.
- Reject image bombs, excessive pixels, oversized files, animation, and malformed metadata.
- Strip EXIF/GPS before storage and processing.
- Treat generated meshes and uploaded files as untrusted; process in restricted containers with CPU/RAM/time/file-count limits and no network unless required.
- Never invoke a shell using user filenames or settings.
- Rate-limit account, IP, upload, generation, status polling, and download endpoints.
- Add email verification, bot protection, and spend caps.
- Content policy and moderation must cover illegal content, weapons, IP infringement, impersonation, and prohibited model-license uses.
- Malware scan downloadable archives and validate archive paths.
- Secrets belong in a managed secret store; rotate worker credentials.
- Do not include local network URLs, absolute paths, prompts, or raw provider responses in customer reports.
- Default source/artifact retention: 30 days for paid users and 24–72 hours for free previews; support immediate project deletion and asynchronous verified purge.
- Maintain encrypted backups for metadata; user content backup policy must match deletion promises.

## 13. Legal/compliance gates

Before taking payment:

- Written model-license review for every codebase and weight artifact.
- GDPR privacy notice, lawful basis, processor list, EU-region choices, access/export/deletion process, and breach procedure.
- Terms requiring upload rights and prohibiting illegal/infringing content.
- Clear AI-generated disclosure and limitations of single-image reconstruction.
- Explicit “not engineering/structural certification” warning for strength-aware guidance.
- Refund/retry policy tied to objective technical failures.
- VAT/OSS and consumer digital-content review for the operator's entity.
- If using Tencent material under separate rights, implement all required end-user notices, use restrictions, provider identity, and non-affiliation wording.

## 14. Observability and operations

### Metrics

- Queue depth/oldest age, jobs by state, success rate, automatic retry rate.
- p50/p95 stage latency, GPU VRAM/utilization/temperature, CPU/RAM/disk.
- Mesh quality-gate failures by model/workflow version.
- Credit reservations/consumption/refunds and webhook failures.
- Conversion, paid attempts per active user, reupload rate, download rate, refund/support rate, contribution margin.

### Alerts

- Worker heartbeat absent for 2 minutes.
- Queue oldest age above target.
- CUDA OOM or repeated model-load failure.
- Disk below 20% free; object upload failures.
- Stripe webhook backlog/signature failures.
- Error-rate or automatic-refund spike after deployment.

### Recovery

- Jobs lease a worker with heartbeat; expired leases return to the queue.
- Reserve credit at enqueue, consume only when GPU work starts, and refund idempotently on eligible failures.
- Upload artifacts to temporary keys and publish via atomic database transaction.
- Pin prior container/workflow/model revision for one-click rollback.
- Nightly metadata backup and quarterly restore test.

## 15. Delivery phases and gates

### Phase 0 — legal and product proof (2–4 days)

- Freeze Hunyuan production work pending rights review.
- Choose the launch model/license and document provenance.
- Interview 10 target users and test pack pricing.
- Define “successful mesh” and retry policy.

**Gate:** model can legally be offered from the EU and users understand the output limitations.

### Phase 1 — repository and reproducible baseline (3–5 days)

- Migrate allowlisted 3DOrienter source to the target monorepo.
- Add pre-commit/ruff/pytest, CI, lockfiles, container builds, fixture licenses.
- Refactor CLI into package service boundaries.
- Preserve the original directory unchanged until the migration is verified and committed.

**Gate:** clean checkout builds and existing tests pass on Linux and Windows/WSL; no secrets or large artifacts in Git history.

### Phase 2 — offline vertical slice (1 week)

- Implement model adapter and one self-hosted backend.
- Build image preflight, mesh validation/repair, scale input, previews.
- Call 3DOrienter library and produce report bundle.
- Add slicer validation for top candidates.

**Gate:** 20-image benchmark completed; 25 sequential jobs without leak; artifacts reproducible by version/seed where the backend permits.

### Phase 3 — private web alpha (1–2 weeks)

- Web upload/status/result UI with 3D preview.
- API, PostgreSQL, Redis queue, object storage, authentication.
- Private worker agent with one-job concurrency, heartbeat, cancellation, cleanup.
- Project/version/reupload flow and objective presets.

**Gate:** no public ComfyUI access; ownership tests; restart recovery; deletion and retention verified.

### Phase 4 — paid beta (1 week)

- Stripe packs, credit ledger, signed webhooks, receipts.
- Free-preview abuse controls, quotas, moderation, legal pages.
- Operations dashboard, alerts, support/admin tools with audit log.

**Gate:** end-to-end payment/refund tests, VAT/legal review, incident runbook, backup restore test.

### Phase 5 — calibrated orientation beta (2–4 weeks in parallel)

- Compare proxy scores with slicer metrics across a representative mesh corpus.
- Print a smaller labeled set and score support removability/surface quality.
- Adjust ranking and publish limitations/confidence.
- Add explicit load cases before exposing strength-aware mode broadly.

**Gate:** documented correlation and regression suite; misleading claims removed.

### Phase 6 — controlled public launch

- Invite batches, queue caps, waitlist when capacity is exhausted.
- Measure reupload, quality-gate, refund, retention, margin, and support load.
- Add cloud overflow only after local economics and demand are known.

**Gate:** two weeks of stable p95 completion, positive contribution margin, and acceptable support/refund rate.

## 16. Launch acceptance criteria

- No unreviewed model/license in production.
- No secret, user artifact, weight file, internal URL, or local path committed or returned in customer reports.
- 100% authorization coverage on project/artifact endpoints.
- Queue delivery and Stripe webhook replays are idempotent.
- Worker restart does not lose jobs or consume credits twice.
- Automatic technical-failure credits are deterministic and auditable.
- User can reupload, compare versions, rerun orientation, and delete the full project.
- Top orientation candidates include actual slicer measurements, not only proxy values.
- Strength-aware mode requires load input and displays a non-certification warning.
- Peak VRAM remains below a defined safe threshold on the 4070 with concurrency one.
- Retention purge and account deletion are demonstrated in staging.
- Backup restoration and rollback are tested.

## 17. Immediate next actions

1. Completed: put this plan and diagram into the target repository.
2. Decide whether to launch with TripoSR or a legally reviewed alternative/API.
3. Run the 20-image feasibility benchmark on the actual 4070.
4. Completed: migrate only the source allowlist; preserve the original folder as a temporary backup.
5. Refactor 3DOrienter into a library and add slicer-backed candidate validation.
6. Build the offline vertical slice before creating accounts, payments, or public networking.

## 18. Primary sources checked

- Hunyuan3D 2.1 repository and VRAM notes: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
- Hunyuan3D 2.1 license: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/LICENSE
- TripoSR repository, license, and VRAM note: https://github.com/VAST-AI-Research/TripoSR
- Stable Fast 3D repository and VRAM note: https://github.com/Stability-AI/stable-fast-3d
- Stable Fast 3D Community License: https://github.com/Stability-AI/stable-fast-3d/blob/main/LICENSE.md
- Microsoft TRELLIS requirements and license notes: https://github.com/microsoft/TRELLIS

License facts can change. Pin and archive the exact license text/revision reviewed for each production release; this document is technical planning, not legal advice.
