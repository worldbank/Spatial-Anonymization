import pytest

from spatial_anonymization.population import find_radius_for_population


def test_population_radius_reaches_threshold():
    """
    Verify that the population-adaptive radius search stops as soon as the
    population threshold is first met, without continuing to larger radii.
    """
    print("\nTesting population-adaptive radius reaches threshold")

    def fake_population(lon, lat, radius_m):
        # 1 person per 100 meters
        return int(radius_m / 100)

    radius = find_radius_for_population(
        population_fn=fake_population,
        lon=35.3,
        lat=-15.4,
        population_threshold=20,
        min_radius_m=500,
        max_radius_m=5000,
        step_m=100,
    )

    print(f"Computed radius: {radius} m")

    # To reach population >= 20 → radius >= 2000
    assert radius >= 2000
    assert radius <= 5000

def test_population_radius_respects_min_radius():
    """
    Verify that the returned radius never falls below the specified minimum
    radius, even when the population threshold is already satisfied.
    """
    print("\nTesting minimum radius enforcement")

    def fake_population(lon, lat, radius_m):
        return 100  # already enough population

    radius = find_radius_for_population(
        population_fn=fake_population,
        lon=35.3,
        lat=-15.4,
        population_threshold=50,
        min_radius_m=1000,
        max_radius_m=5000,
        step_m=100,
    )

    print(f"Computed radius: {radius} m")

    assert radius == 1000


def test_population_radius_failure_when_threshold_unreachable():
    """
    Verify that the algorithm fails cleanly when the population threshold
    cannot be reached within the maximum allowed radius.
    """
    print("\nTesting failure when population threshold is unreachable")

    def fake_population(lon, lat, radius_m):
        return 5  # never increases

    with pytest.raises(ValueError):
        find_radius_for_population(
            population_fn=fake_population,
            lon=35.3,
            lat=-15.4,
            population_threshold=50,
            min_radius_m=500,
            max_radius_m=2000,
            step_m=100,
        )
