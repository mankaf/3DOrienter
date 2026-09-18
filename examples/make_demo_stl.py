"""Generate a simple overhanging bracket for trying the CLI."""

from pathlib import Path

import trimesh


def main() -> None:
    base = trimesh.creation.box(extents=(40, 30, 4))
    base.apply_translation((0, 0, 2))

    wall = trimesh.creation.box(extents=(4, 30, 30))
    wall.apply_translation((-18, 0, 15))

    shelf = trimesh.creation.box(extents=(24, 30, 4))
    shelf.apply_translation((-8, 0, 28))

    bracket = trimesh.util.concatenate([base, wall, shelf])
    output = Path(__file__).with_name("demo_bracket.stl")
    bracket.export(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
