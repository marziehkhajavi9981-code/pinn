import numpy as np


for path in ("outputs/field_predictions.npz", "u_stack.npz"):
    with np.load(path) as archive:
        print(f"\nFILE {path}")
        print("keys", archive.files)
        for key in archive.files:
            array = archive[key]
            print(
                key,
                "shape", array.shape,
                "dtype", array.dtype,
                "min", float(np.nanmin(array)) if array.size else None,
                "max", float(np.nanmax(array)) if array.size else None,
                "mean", float(np.nanmean(array)) if array.size else None,
            )
