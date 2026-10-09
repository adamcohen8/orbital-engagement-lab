"""Lazy optional Numba kernels. Never use fastmath or parallel reductions."""

import math

import numpy as np

from sim.acceleration.optional import njit_or_identity
from sim.pro_perturbations._numeric import solid_harmonics_numeric

solid_harmonics_kernel = njit_or_identity(cache=True)(solid_harmonics_numeric)


@njit_or_identity(cache=True)
def tidal_acceleration_kernel(r, c, s, mu, radius, normalization):
    distance = np.linalg.norm(r)
    h, g = solid_harmonics_kernel(r / distance, normalization, True)
    acceleration = np.zeros(3)
    for n in range(2, len(c)):
        scale = mu / distance**2 * (radius / distance) ** n
        for m in range(n + 1):
            for k in range(3):
                gradient = g[n, m, k] - (2 * n + 1) * h[n, m] * r[k] / distance
                acceleration[k] += scale * (c[n, m] * gradient.real + s[n, m] * gradient.imag)
    return acceleration


@njit_or_identity(cache=True)
def schwarzschild_kernel(x, mu, radius):
    r, v = x[:3], x[3:]
    return mu / (299792.458**2 * radius**3) * ((4 * mu / radius - v @ v) * r + 4 * (r @ v) * v)


@njit_or_identity(cache=True)
def radiation_kernel(r, sun, elapsed, radius, nodes, weights, cp, sp):
    distance = np.linalg.norm(r)
    sun_distance = np.linalg.norm(sun)
    radial = r / distance
    east = np.cross(np.array([0.0, 0.0, 1.0]), radial)
    east_norm = np.linalg.norm(east)
    if east_norm < 1e-12:
        east = np.array([0.0, 1.0, 0.0])
    else:
        east /= east_norm
    north = np.cross(radial, east)
    cos_limb = math.sqrt(max(0.0, 1 - (radius / distance) ** 2))
    width = (radius / distance) ** 2 / (1 + cos_limb)
    season = math.cos(2 * math.pi * elapsed / (365.25 * 86400))
    a, ir = np.zeros(3), np.zeros(3)
    order = len(nodes)
    sun_unit = sun / sun_distance
    direction = np.empty(3)
    normal = np.empty(3)
    for i in range(order):
        cosine = 1 - width * (1 - nodes[i]) / 2
        sine = math.sqrt(max(0.0, 1 - cosine * cosine))
        root = math.sqrt(max(0.0, radius**2 - distance**2 * (1 - cosine * cosine)))
        length = (distance**2 - radius**2) / (distance * cosine + root)
        weight = weights[i] * (width / 2) * (2 * math.pi / (4 * order))
        for j in range(4 * order):
            for k in range(3):
                direction[k] = cosine * radial[k] + sine * (cp[j] * east[k] + sp[j] * north[k])
                normal[k] = (r[k] - length * direction[k]) / radius
            lat = normal[2]
            p2 = 0.5 * (3 * lat**2 - 1)
            albedo = 0.34 + 0.10 * season * lat + 0.29 * p2
            emissivity = 0.68 - 0.07 * season * lat - 0.18 * p2
            incidence = 0.0
            for k in range(3):
                incidence += normal[k] * sun_unit[k]
            incidence = max(0.0, incidence)
            for k in range(3):
                wd = weight * direction[k]
                a[k] += wd * (albedo * incidence)
                ir[k] += wd * emissivity
    pressure = 4.5606e-6 * (149597870.691 / sun_distance) ** 2
    return a * pressure / math.pi, ir * pressure / (4 * math.pi)
