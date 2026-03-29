"""
Radius selection strategies for spatial anonymization.

This module defines how far a point is allowed to move from its original
location. It does NOT check privacy or realism constraints. It only
determines candidate displacement distances.

Implemented strategies:
- DHS-style random displacement (area-uniform over disk, via donut with min=0)
- Donut geomasking (min → max distance, area-uniform)
- Population-adaptive radius selection (minimum radius satisfying a
  population threshold)

These radius values are later combined with spatial constraints
(e.g. admin boundaries, population checks) in SurveyDisplacer.
"""


# radius.py
import math
import random
from typing import Callable, Optional



def sample_donut_radius(min_distance_m: float, max_distance_m: float) -> float:
    """Sample donut geomasking radius uniformly by area.

    Args:
        min_distance_m (float): Minimum displacement distance in meters.
        max_distance_m (float): Maximum displacement distance in meters.

    Returns:
        float: Displacement radius in meters.
    """
    u = random.random()
    return math.sqrt(
        u * (max_distance_m**2 - min_distance_m**2) + min_distance_m**2
    )


def find_population_adaptive_radius(
    *,
    population_check_fn: Callable[[float], bool],
    step_m: float = 250.0,
    max_radius_m: Optional[float] = None,
) -> Optional[float]:
    """Find minimum radius that satisfies population constraint.

    Args:
        population_check_fn (Callable[[float], bool]):
            Function that returns True if population threshold is met
            for a given radius.
        step_m (float): Radius increment in meters.
        max_radius_m (Optional[float]): Optional upper bound.

    Returns:
        Optional[float]: Radius in meters or None if not found.
    """
    radius = step_m
    while max_radius_m is None or radius <= max_radius_m:
        if population_check_fn(radius):
            return radius
        radius += step_m
    return None
