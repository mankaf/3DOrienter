# Headless slicer validation setup

`OrcaFamilySlicerBackend` (`src/orienter3d/service/slicing.py`) drives a locally installed
ElegooSlicer/OrcaSlicer/BambuStudio-family executable in headless CLI mode to produce real
print-time, filament, and support estimates for the oriented STL, instead of relying on the
geometric proxies alone. Everything below was verified against a real ElegooSlicer 1.5.3.5
install (`elegoo-slicer.exe`) on Windows; the CLI surface is shared across the OrcaSlicer family,
so other forks should behave the same way, but re-verify before trusting a new vendor build.

## Critical: only `--slice` mode is headless

Running the executable with a bare `--help` (or any invocation that doesn't include `--slice`)
boots the **full GUI application**, including its printer-network stack. On a machine with a
configured network printer this silently launches a background process that repeatedly tries to
reconnect to that printer's LAN address every ~10 seconds until killed. Never invoke the
executable without `--slice` and an input file; that combination is what routes execution through
the headless pipeline instead of `GUI_App::OnInit`.

## Confirmed CLI invocation

```
<exe> <input.stl> \
  --load-settings "<process.json>;<machine.json>" \
  --load-filaments <filament.json> \
  --slice 0 \
  --export-3mf <output.gcode.3mf>
```

- `--load-settings` takes a semicolon-joined `process;machine` pair, both from the vendor's
  resolved preset tree (see below).
- `--slice 0` slices all plates; a single input STL produces exactly one plate. `--slice 1` was
  not reliably reproducible in testing and isn't used.
- Invoke via `subprocess.run([...])` with an explicit argv list (`shell=False`). Building the
  command line as a single string (e.g. via PowerShell's `Start-Process -ArgumentList`) breaks on
  paths containing spaces — vendor preset filenames routinely contain spaces (`"0.20mm Standard
  @Elegoo CC 0.4 nozzle.json"`).

## Finding the right preset files

Vendor presets are not one file per printer model. Each vendor ships a top-level index
(`resources/profiles/<Vendor>.json`) mapping a human-readable preset name to a `sub_path`, e.g.:

```json
{ "name": "Elegoo Centauri Carbon 0.4 nozzle", "sub_path": "machine/ECC/Elegoo Centauri Carbon 0.4 nozzle.json" }
{ "name": "0.20mm Standard @Elegoo CC 0.4 nozzle", "sub_path": "process/ECC/0.20mm Standard @Elegoo CC 0.4 nozzle.json" }
{ "name": "Elegoo PLA @ECC", "sub_path": "filament/ECC/Elegoo PLA @ECC.json" }
```

Search that index for the target printer name rather than guessing a filename. The raw files
under `machine/`, `process/`, and `filament/` at the top level are shared base templates
(`fdm_machine_common.json`, etc.) resolved via `inherits` chains — they are not directly usable
presets on their own.

It's safe to point `--load-settings`/`--load-filaments` at the factory files under the install's
`resources/profiles/` tree (read-only, no network fields). Avoid the user's own saved printer
profiles under `%APPDATA%/<Vendor>/printers/*.json` for automation — those carry the user's saved
network printer IP and are what triggers GUI-style reconnect attempts if accidentally loaded
through a non-headless path.

## Known gotcha: point-contact geometry breaks slicing

A mesh whose lowest point is a single vertex (e.g. an icosphere resting on its pole) reliably
fails with a generic, unhelpful error:

```
Slic3r::CLI::run found error, exit
STDERR: Errors
```

The debug log (`--debug 5 --logfile <path>`) shows the pipeline running normally through wall/
infill generation, then failing at "Generating skirt & brim" with `found slicing or export error
for partplate 1` and no further detail. A flat-bottomed test shape (a box) slices cleanly with the
identical command. If a real orientation candidate produces this failure, treat it as a quality
signal — the candidate has effectively zero first-layer contact area — rather than assuming the
slicer integration itself is broken.

## Output format: `Metadata/slice_info.config`

`--export-3mf` writes a `.gcode.3mf` (a zip). `_parse_slice_info` in `slicing.py` reads
`Metadata/slice_info.config`, an XML document with the schema below (fields actually used are
noted):

```xml
<config>
  <header>
    <header_item key="X-BBL-Client-Name" value="ElegooSlicer"/>   <!-- -> engine -->
    <header_item key="X-BBL-Client-Version" value="01.05.03.05"/> <!-- -> engine_version -->
  </header>
  <plate>
    <metadata key="prediction" value="365"/>        <!-- print time, seconds -->
    <metadata key="support_used" value="false"/>
    <filament id="1" used_m="0.48" used_g="0.00" .../>  <!-- summed across all <filament> tags -->
  </plate>
</config>
```

`prediction` cross-validated exactly against the gcode header comment
(`; estimated printing time (normal mode) = 6m 5s` = 365s).

**`used_g` can legitimately be `0.00`** even for a multi-meter print, if the loaded filament
profile has no `filament_density` set. This was observed on the stock `Elegoo PLA @ECC` profile.
`used_m` (converted to mm in `SlicingReport.filament_used_mm`) is the more reliable field until
that's confirmed fixed upstream; don't treat `filament_used_g == 0` as a parsing bug on its own.

## CLI usage

```bash
3dorienter-pipeline input.png \
  --backend mock \
  --slicer-exe "C:/Program Files/ElegooSlicer/elegoo-slicer.exe" \
  --slicer-process-profile "C:/Program Files/ElegooSlicer/resources/profiles/Elegoo/process/ECC/0.20mm Standard @Elegoo CC 0.4 nozzle.json" \
  --slicer-machine-profile "C:/Program Files/ElegooSlicer/resources/profiles/Elegoo/machine/ECC/Elegoo Centauri Carbon 0.4 nozzle.json" \
  --slicer-filament-profile "C:/Program Files/ElegooSlicer/resources/profiles/Elegoo/filament/ECC/Elegoo PLA @ECC.json"
```

All four `--slicer-*` flags are required together, or all omitted; slicing stays fully opt-in.
