import math

import numba as nb
import numpy as np


@nb.njit()
def binary_scalar_radii(coords, types, axis_x, axis_y, axis_z, radii, out):
    assert coords.ndim == 2
    assert coords.shape[1] == 3

    assert types.ndim == 1
    assert types.shape[0] == coords.shape[0]

    assert axis_x.ndim == 1
    assert axis_y.ndim == 1
    assert axis_z.ndim == 1

    assert np.isscalar(radii)

    assert out.ndim == 4
    assert out.shape[1] == axis_x.shape[0]
    assert out.shape[2] == axis_y.shape[0]
    assert out.shape[3] == axis_z.shape[0]

    radii_sq = radii**2
    for v in range(coords.shape[0]):
        c = types[v]
        px, py, pz = coords[v, 0], coords[v, 1], coords[v, 2]
        for x, ax in enumerate(axis_x):
            dx_sq = (px - ax) ** 2
            for y, ay in enumerate(axis_y):
                dxdy_sq = dx_sq + (py - ay) ** 2
                for z, az in enumerate(axis_z):
                    distance_sq = dxdy_sq + (pz - az) ** 2
                    if distance_sq < radii_sq:
                        out[c, x, y, z] += 1
    return out


@nb.njit()
def gaussian_scalar_radii(coords, types, axis_x, axis_y, axis_z, radii, sigma, out):
    assert coords.ndim == 2
    assert coords.shape[1] == 3

    assert types.ndim == 1
    assert types.shape[0] == coords.shape[0]

    assert axis_x.ndim == 1
    assert axis_y.ndim == 1
    assert axis_z.ndim == 1

    assert np.isscalar(radii)
    assert np.isscalar(sigma)

    assert out.ndim == 4
    assert out.shape[1] == axis_x.shape[0]
    assert out.shape[2] == axis_y.shape[0]
    assert out.shape[3] == axis_z.shape[0]

    radii_sq = radii**2
    multiplier = -0.5 / radii_sq / sigma**2
    for v in range(coords.shape[0]):
        c = types[v]
        px, py, pz = coords[v, 0], coords[v, 1], coords[v, 2]
        for x, ax in enumerate(axis_x):
            dx_sq = (px - ax) ** 2
            for y, ay in enumerate(axis_y):
                dxdy_sq = dx_sq + (py - ay) ** 2
                for z, az in enumerate(axis_z):
                    distance_sq = dxdy_sq + (pz - az) ** 2
                    if distance_sq < radii_sq:
                        out[c, x, y, z] += math.exp(multiplier * distance_sq)
    return out


@nb.njit()
def binary_atom_wise_radii(coords, types, axis_x, axis_y, axis_z, radii, out):
    assert coords.ndim == 2
    assert coords.shape[1] == 3

    assert types.ndim == 1
    assert types.shape[0] == coords.shape[0]

    assert axis_x.ndim == 1
    assert axis_y.ndim == 1
    assert axis_z.ndim == 1

    assert radii.ndim == 1
    assert radii.shape[0] == coords.shape[0]

    assert out.ndim == 4
    assert out.shape[1] == axis_x.shape[0]
    assert out.shape[2] == axis_y.shape[0]
    assert out.shape[3] == axis_z.shape[0]

    for v in range(coords.shape[0]):
        c = types[v]
        radii_sq = radii[v] ** 2
        px, py, pz = coords[v, 0], coords[v, 1], coords[v, 2]
        for x, ax in enumerate(axis_x):
            dx_sq = (px - ax) ** 2
            for y, ay in enumerate(axis_y):
                dxdy_sq = dx_sq + (py - ay) ** 2
                for z, az in enumerate(axis_z):
                    distance_sq = dxdy_sq + (pz - az) ** 2
                    if distance_sq < radii_sq:
                        out[c, x, y, z] += 1
    return out


@nb.njit()
def gaussian_atom_wise_radii(coords, types, axis_x, axis_y, axis_z, radii, sigma, out):
    assert coords.ndim == 2
    assert coords.shape[1] == 3

    assert types.ndim == 1
    assert types.shape[0] == coords.shape[0]

    assert axis_x.ndim == 1
    assert axis_y.ndim == 1
    assert axis_z.ndim == 1

    assert radii.ndim == 1
    assert np.isscalar(sigma)

    assert out.ndim == 4
    assert out.shape[1] == axis_x.shape[0]
    assert out.shape[2] == axis_y.shape[0]
    assert out.shape[3] == axis_z.shape[0]

    sigma_sq = sigma**2
    for v in range(coords.shape[0]):
        c = types[v]
        radii_sq = radii[v] ** 2
        multiplier = -0.5 / radii_sq / sigma_sq
        px, py, pz = coords[v, 0], coords[v, 1], coords[v, 2]
        for x, ax in enumerate(axis_x):
            dx_sq = (px - ax) ** 2
            for y, ay in enumerate(axis_y):
                dxdy_sq = dx_sq + (py - ay) ** 2
                for z, az in enumerate(axis_z):
                    distance_sq = dxdy_sq + (pz - az) ** 2
                    if distance_sq < radii_sq:
                        out[c, x, y, z] += math.exp(multiplier * distance_sq)
    return out


@nb.njit()
def binary_channel_wise_radii(coords, types, axis_x, axis_y, axis_z, radii, out):
    assert coords.ndim == 2
    assert coords.shape[1] == 3

    assert types.ndim == 1
    assert types.shape[0] == coords.shape[0]

    assert axis_x.ndim == 1
    assert axis_y.ndim == 1
    assert axis_z.ndim == 1

    assert radii.ndim == 1

    assert out.ndim == 4
    assert out.shape[1] == axis_x.shape[0]
    assert out.shape[2] == axis_y.shape[0]
    assert out.shape[3] == axis_z.shape[0]

    for v in range(coords.shape[0]):
        c = types[v]
        radii_sq = radii[c] ** 2
        px, py, pz = coords[v, 0], coords[v, 1], coords[v, 2]
        for x, ax in enumerate(axis_x):
            dx_sq = (px - ax) ** 2
            for y, ay in enumerate(axis_y):
                dxdy_sq = dx_sq + (py - ay) ** 2
                for z, az in enumerate(axis_z):
                    distance_sq = dxdy_sq + (pz - az) ** 2
                    if distance_sq < radii_sq:
                        out[c, x, y, z] += 1
    return out


@nb.njit()
def gaussian_channel_wise_radii(coords, types, axis_x, axis_y, axis_z, radii, sigma, out):
    assert coords.ndim == 2
    assert coords.shape[1] == 3

    assert types.ndim == 1
    assert types.shape[0] == coords.shape[0]

    assert axis_x.ndim == 1
    assert axis_y.ndim == 1
    assert axis_z.ndim == 1

    assert radii.ndim == 1
    assert np.isscalar(sigma)

    assert out.ndim == 4
    assert out.shape[1] == axis_x.shape[0]
    assert out.shape[2] == axis_y.shape[0]
    assert out.shape[3] == axis_z.shape[0]

    sigma_sq = sigma**2
    for v in range(coords.shape[0]):
        c = types[v]
        radii_sq = radii[c] ** 2
        multiplier = -0.5 / radii_sq / sigma_sq
        px, py, pz = coords[v, 0], coords[v, 1], coords[v, 2]
        for x, ax in enumerate(axis_x):
            dx_sq = (px - ax) ** 2
            for y, ay in enumerate(axis_y):
                dxdy_sq = dx_sq + (py - ay) ** 2
                for z, az in enumerate(axis_z):
                    distance_sq = dxdy_sq + (pz - az) ** 2
                    if distance_sq < radii_sq:
                        out[c, x, y, z] += math.exp(multiplier * distance_sq)
    return out
