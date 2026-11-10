Meshing tool for the generation of structured C-grids of (half) lifting surfaces in 2D and 3D using the blockMesh utility of OpenFOAM

## Usage

Run an example script from the directory where the blockMesh lists should go (e.g. your OpenFOAM case):

```
python /path/to/mshGen/examples/generate_mesh_NACA0012_AIRFOIL.py
```

The scripts locate `mshGen/meshGeneration` themselves (no `PYTHONPATH` needed). Output is written to
`./output_files`; airfoil, wing and default files are taken from `mshGen/meshGeneration/`.
Include the generated lists in `system/blockMeshDict` (see `openFoamMesh/system/blockMeshDict.*`).

### Inputs

- `params = [dim, COG, slice_locations, layer_resolution, layer_grading, layer_aileron, mesh_scale, mesh_level, skip_layer]`
- `flags = [WRITE, PRINT, PLOT, DEFAULT, SPLIT, ARTIFICIOUS]`
- `layer_aileron`: `None`, a layer index, or a list of consecutive layer indices (e.g. `[2, 3]`).
  The aileron cut-out may include the tip layer.
- `DEFAULT=True` writes the default point/resolution files to `meshGeneration/default_files`
  (overwriting existing files of the same case name); `DEFAULT=False` reads them.
- `SPLIT=True` writes the wing surface as `wingUp` / `wingDown`; with `ARTIFICIOUS=True` the first
  `skip_layer` wing layers go to a separate `extra` patch.

### Wing files (`meshGeneration/wings/`)

One line per spanwise station: `station  y  chord  twist(beta)  dx  [dz]`.
The optional 6th column `dz` shifts the section vertically (dihedral); without it `dz = 0`.
Beyond the tip, the tip values are used.
