"""Recover a Reduced-3DGS SH3 half PLY from an existing quantised PLY.

Only converts position and embedded codebook floats; preserves quantised IDs.
The output must be a new file outside the project output directory. No CUDA,
training, clustering, codebook.pt, or original output modification is needed.
"""
import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

PROJECT_OUTPUT = Path(__file__).resolve().parent.parent / 'output'


def _schema(degree):
    return (['x', 'y', 'z'] + [f'f_dc_{i}' for i in range(3)] +
            [f'f_rest_{i}' for i in range(((degree + 1) ** 2 - 1) * 3)] +
            ['opacity'] + [f'scale_{i}' for i in range(3)] +
            [f'rot_{i}' for i in range(4)])


def _half_bits(values, label):
    if not np.isfinite(values).all() or np.any(np.abs(values) > 65504):
        raise ValueError(f'{label} contains non-finite or out-of-range float16 values')
    return np.asarray(values, dtype=np.float16).view(np.int16)


def _transform(ply):
    if ply.text or ply.byte_order != '<':
        raise ValueError('only binary little-endian PLY is supported')
    names = [f'vertex_{i}' for i in range(4)] + ['codebook_centers']
    if [e.name for e in ply.elements] != names:
        raise ValueError('expected vertex_0..vertex_3 and codebook_centers in order')
    result = []
    for degree in range(4):
        el = ply.elements[degree]
        fields = _schema(degree)
        if list(el.data.dtype.names or ()) != fields:
            raise ValueError(f'{el.name}: unexpected properties')
        for name in fields:
            dt = el.data.dtype.fields[name][0]
            if dt.kind != ('f' if name in ('x', 'y', 'z') else 'u') or dt.itemsize != (4 if name in ('x', 'y', 'z') else 1):
                raise ValueError(f'{el.name}.{name}: unexpected type')
        out = np.empty(el.count, dtype=[(name, '<i2' if name in ('x', 'y', 'z') else 'u1') for name in fields])
        for name in fields:
            out[name] = _half_bits(el[name], f'{el.name}.{name}') if name in ('x', 'y', 'z') else el[name]
        result.append(PlyElement.describe(out, el.name))
    el = ply.elements[-1]
    fields = ['features_dc'] + [f'features_rest_{i}' for i in range(15)] + ['opacity', 'scaling', 'rotation_re', 'rotation_im']
    if el.count != 256 or list(el.data.dtype.names or ()) != fields:
        raise ValueError('expected 256 rows and standard SH3 codebook properties')
    if any(el.data.dtype.fields[n][0].kind != 'f' or el.data.dtype.fields[n][0].itemsize != 4 for n in fields):
        raise ValueError('expected float32 codebook properties')
    out = np.empty(256, dtype=[(name, '<i2') for name in fields])
    for name in fields:
        out[name] = _half_bits(el[name], f'codebook_centers.{name}')
    result.append(PlyElement.describe(out, el.name))
    return PlyData(result, byte_order='<')


def recover(source, destination):
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if destination == PROJECT_OUTPUT.resolve() or PROJECT_OUTPUT.resolve() in destination.parents:
        raise ValueError('destination must be outside the project output directory')
    if destination == source:
        raise ValueError('source and destination must differ')
    ply = PlyData.read(str(source))
    converted = _transform(ply)
    # Exclusive creation protects all existing files, including the input.
    with destination.open('xb') as stream:
        try:
            converted.write(stream)
            stream.flush()
        except BaseException:
            stream.close()
            destination.unlink(missing_ok=True)
            raise
    # Read back and compare every field, without invoking rendering or evaluation.
    try:
        actual = PlyData.read(str(destination))
        if [e.name for e in actual.elements] != [e.name for e in converted.elements]:
            raise ValueError('written PLY element names differ')
        for expected, found in zip(converted.elements, actual.elements):
            if expected.count != found.count or expected.data.dtype != found.data.dtype or not np.array_equal(expected.data, found.data):
                raise ValueError(f'written PLY differs in {expected.name}')
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return sum(e.count for e in ply.elements[:-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='original float32 quantised PLY (read-only)')
    parser.add_argument('destination', type=Path, help='new PLY outside project output; parent must exist')
    args = parser.parse_args()
    count = recover(args.source, args.destination)
    print(f'Recovered {count} primitives to {args.destination}; training status unchanged.')


if __name__ == '__main__':
    main()
