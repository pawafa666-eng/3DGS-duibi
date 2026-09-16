# Bundled 3DGS runtime

This directory contains the verified 3DGS training and rendering source used
by the comparison runner. It is intentionally kept inside the comparison
repository so a clean checkout does not depend on a machine-specific `repo`
directory.

The runtime includes the CUDA extensions under `submodules/`. Install them in
the active Conda environment before running the comparison:

```powershell
python -m pip install .\01_3DGS\runtime\submodules\diff-gaussian-rasterization
python -m pip install .\01_3DGS\runtime\submodules\simple-knn
```

The bundled loader matches this experiment's custom `points3D.bin` records,
which contain XYZ, RGB, normals, error, and track length fields. It also reads
`is_ground.txt` when that file is present. Dataset files and generated outputs
are not part of this source bundle.
