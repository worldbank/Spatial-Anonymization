from build.lib.spatial_anonymization.point_displacement import distance_latlon_points
from spatial_anonymization.geospatial.geo_utils import (
    distance_points,
    displace_point_geodesic,
)

# ---------------------------------------------------------
# Malawi reference locations (real, public places)
# ---------------------------------------------------------

ZOMBA_NSO = (35.323140960899856, -15.376682983197496)
CHANCELLOR_COLLEGE = (35.33540190374707, -15.387781865785113)
NKHOTAKOTA_LODGE = (34.32717051655966, -13.049378200732006)
ZOMBA_OUTSKIRTS = (35.29666767241161, -15.424803318797395)


# ---------------------------------------------------------
# Distance tests
# ---------------------------------------------------------

def test_distance_zero_same_location():
    print("\nTesting distance between identical locations (Zomba NSO)")
    dist = distance_points(ZOMBA_NSO, ZOMBA_NSO)
    print(f"Computed distance: {dist:.6f} km")

    assert dist == 0.0


def test_distance_zomba_nso_to_chanco_reasonable():
    print("\nTesting distance: Zomba NSO → Chancellor College (Chanco)")
    dist_km = distance_points(ZOMBA_NSO, CHANCELLOR_COLLEGE)
    print(f"Computed distance: {dist_km:.2f} km (expected ~2–4 km)")

    assert dist_km > 1
    assert dist_km < 5


def test_distance_zomba_nso_to_zomba_outskirts_reasonable():
    print("\nTesting distance: Zomba NSO → Zomba outskirts (5–10 km reference)")
    dist_km = distance_points(ZOMBA_NSO, ZOMBA_OUTSKIRTS)
    print(f"Computed distance: {dist_km:.2f} km (expected ~5–10 km)")

    assert dist_km > 4
    assert dist_km < 12


def test_distance_zomba_to_nkhotakota_reasonable():
    print("\nTesting distance: Zomba NSO → Nkhotakota Pottery Lodge")
    dist_km = distance_points(ZOMBA_NSO, NKHOTAKOTA_LODGE)
    print(f"Computed distance: {dist_km:.1f} km (expected ~200–300 km)")

    assert dist_km > 200
    assert dist_km < 350


# ---------------------------------------------------------
# Displacement with bearing test
# ---------------------------------------------------------
def test_displace_point_geodesic_distance():
    new_lon, new_lat = displace_point_geodesic(
        lon=ZOMBA_NSO[0],
        lat=ZOMBA_NSO[1],
        distance_m=5000,
        bearing=90
    )

    dist = distance_points(
        ZOMBA_NSO,
        (new_lon, new_lat)
    )

    print(f"Expected ~5km, got {dist:.2f} km")
    assert 4.5 < dist < 5.5
