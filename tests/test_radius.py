import numpy as np

from spatial_anonymization.radius import (
    sample_dhs_radius,
    sample_donut_radius,
)


# =========================================================
# DHS-STYLE RADIUS TESTS
# =========================================================

def test_dhs_radius_within_bounds():
    print("\nTesting DHS radius bounds")
    max_dist = 5000  # meters

    samples = [sample_dhs_radius(max_dist) for _ in range(1000)]

    print(f"Min sampled distance: {min(samples):.2f} m")
    print(f"Max sampled distance: {max(samples):.2f} m")

    assert min(samples) >= 0
    assert max(samples) <= max_dist


def test_dhs_radius_allows_small_distances():
    print("\nTesting DHS allows very small displacements")
    max_dist = 5000

    samples = [sample_dhs_radius(max_dist) for _ in range(1000)]

    small_count = sum(d < 100 for d in samples)  # <100 m
    print(f"Number of samples < 100 m: {small_count}")

    assert small_count > 0


# =========================================================
# DONUT RADIUS TESTS
# =========================================================

def test_donut_radius_within_bounds():
    print("\nTesting donut radius bounds")
    min_dist = 2000
    max_dist = 5000

    samples = [sample_donut_radius(min_dist, max_dist) for _ in range(1000)]

    print(f"Min sampled distance: {min(samples):.2f} m")
    print(f"Max sampled distance: {max(samples):.2f} m")

    assert min(samples) >= min_dist
    assert max(samples) <= max_dist


def test_donut_radius_never_below_min():
    print("\nTesting donut radius never produces < min_distance")
    min_dist = 2000
    max_dist = 5000

    samples = [sample_donut_radius(min_dist, max_dist) for _ in range(1000)]

    violations = [d for d in samples if d < min_dist]
    print(f"Violations found: {len(violations)}")

    assert len(violations) == 0
