"""Derived orbital quantities (all DERIVED from real mean elements, not observed)."""
from __future__ import annotations
from dataclasses import dataclass
import math

MU = 398600.4418       # km^3/s^2 (WGS84/EGM96)
R_EARTH = 6378.137     # km, equatorial


@dataclass(frozen=True)
class DerivedElements:
    semi_major_axis_km: float
    period_min: float
    perigee_alt_km: float
    apogee_alt_km: float
    note: str = "Derived from mean motion via Kepler's third law; mean (Kozai) elements, not osculating."


def derive(mean_motion_rev_day: float, eccentricity: float) -> DerivedElements:
    n = mean_motion_rev_day * 2 * math.pi / 86400.0     # rad/s
    a = (MU / n ** 2) ** (1.0 / 3.0)
    return DerivedElements(a, 1440.0 / mean_motion_rev_day,
                           a * (1 - eccentricity) - R_EARTH,
                           a * (1 + eccentricity) - R_EARTH)
