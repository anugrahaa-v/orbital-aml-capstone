"""Coordinate frames. SGP4 outputs TEME (km, km/s). Conversions are explicit.

TEME -> ECEF (PEF/ITRF approx): rotate about Z by GMST (IAU-1982), subtract
omega x r for velocity. ASSUMPTIONS (documented limitations): UT1 ~= UTC
(|UT1-UTC| < 0.9 s, <= ~0.4 km Earth-fixed position error), polar motion and
equation-of-equinoxes terms ignored.
"""
from __future__ import annotations
import numpy as np
from sgp4.propagation import gstime

WGS84_A = 6378.137                    # km
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)
OMEGA_EARTH = 7.2921158553e-5         # rad/s


def gmst(jd_ut1: float) -> float:
    return gstime(jd_ut1)


def teme_to_ecef(r_teme, v_teme, jd_ut1: float):
    th = gmst(jd_ut1)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    r = R @ np.asarray(r_teme, float)
    w = np.array([0.0, 0.0, OMEGA_EARTH])
    v = R @ np.asarray(v_teme, float) - np.cross(w, r)
    return r, v


def ecef_to_geodetic(r_ecef):
    """ECEF (km) -> (lat_deg, lon_deg, alt_km), WGS84, iterative."""
    x, y, z = map(float, r_ecef)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    alt = 0.0
    for _ in range(10):
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - N if abs(np.cos(lat)) > 1e-12 else abs(z) / np.sin(lat) - N * (1 - WGS84_E2)
        new = np.arctan2(z, p * (1 - WGS84_E2 * N / (N + alt)))
        if abs(new - lat) < 1e-13:
            lat = new
            break
        lat = new
    return float(np.degrees(lat)), float(np.degrees(lon)), float(alt)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    return np.array([(N + alt_km) * np.cos(lat) * np.cos(lon),
                     (N + alt_km) * np.cos(lat) * np.sin(lon),
                     (N * (1 - WGS84_E2) + alt_km) * np.sin(lat)])


def teme_to_ecef_many(r_teme, jd_ut1):
    """r_teme (T,3), jd_ut1 (T,) -> ECEF positions (T,3). Same assumptions as teme_to_ecef."""
    th = np.array([gstime(j) for j in np.atleast_1d(jd_ut1)])
    c, s = np.cos(th), np.sin(th)
    x, y, z = r_teme[:, 0], r_teme[:, 1], r_teme[:, 2]
    return np.stack([c * x + s * y, -s * x + c * y, z], axis=1)


def ecef_to_geodetic_many(r):
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    lon = np.arctan2(y, x); p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - WGS84_E2)); alt = np.zeros_like(p)
    for _ in range(8):
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - WGS84_E2 * N / (N + alt)))
    return np.degrees(lat), np.degrees(lon), alt
