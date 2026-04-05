"""
Quick test script for download functionality.

This script tests the download functions in InputsManager:
- Admin boundaries from geoBoundaries/GADM
- Population raster from WorldPop
- Water bodies from OpenStreetMap
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spatial_anonymization.inputs_manager import InputsManager
import tempfile


def test_admin_download():
    """Test downloading admin boundaries."""
    print("\n" + "="*60)
    print("TEST 1: Downloading Admin Boundaries")
    print("="*60)
    
    try:
        # Test with Malawi (small country, fast download)
        inputs = InputsManager(
            country_iso3="MWI",
            admin_level=2,
            auto_download=True,
        )
        
        admin_gdf = inputs.get_admin_gdf()
        print(f"✓ Successfully downloaded admin boundaries")
        print(f"  - Number of features: {len(admin_gdf)}")
        print(f"  - CRS: {admin_gdf.crs}")
        print(f"  - Columns: {list(admin_gdf.columns)[:5]}...")
        print(f"  - Bounds: {admin_gdf.total_bounds}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to download admin boundaries: {e}")
        return False


def test_population_download():
    """Test downloading WorldPop raster."""
    print("\n" + "="*60)
    print("TEST 2: Downloading Population Raster")
    print("="*60)
    
    try:
        # Create a temp directory for downloads
        download_dir = tempfile.mkdtemp()
        print(f"Using download directory: {download_dir}")
        
        # Test with Malawi
        inputs = InputsManager(
            country_iso3="MWI",
            admin_level=0,
            auto_download=True,
            download_dir=download_dir,
        )
        
        # Download population raster
        raster_path = inputs.download_population_raster(year=2020)
        
        print(f"✓ Successfully downloaded population raster")
        print(f"  - Path: {raster_path}")
        print(f"  - File exists: {Path(raster_path).exists()}")
        print(f"  - File size: {Path(raster_path).stat().st_size / 1024 / 1024:.2f} MB")
        
        # Try opening it
        dataset = inputs.get_population_dataset()
        if dataset:
            print(f"  - Raster dimensions: {dataset.width} x {dataset.height}")
            print(f"  - CRS: {dataset.crs}")
            dataset.close()
        
        return True
    except Exception as e:
        print(f"✗ Failed to download population raster: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_water_download():
    """Test downloading water bodies from OSM."""
    print("\n" + "="*60)
    print("TEST 3: Downloading Water Bodies from OSM")
    print("="*60)
    
    try:
        # Test with Malawi - first get admin boundaries
        inputs = InputsManager(
            country_iso3="MWI",
            admin_level=0,  # Country level for faster download
            auto_download=True,
        )
        
        # Download water bodies
        water_gdf = inputs.download_water_bodies()
        
        if water_gdf is not None:
            print(f"✓ Successfully downloaded water bodies")
            print(f"  - Number of features: {len(water_gdf)}")
            print(f"  - CRS: {water_gdf.crs}")
            print(f"  - Geometry types: {water_gdf.geometry.type.value_counts().to_dict()}")
            print(f"  - Sample tags: {water_gdf.iloc[0].to_dict() if len(water_gdf) > 0 else 'N/A'}")
        else:
            print("⚠ No water bodies found (this might be normal for some areas)")
        
        return True
    except Exception as e:
        print(f"✗ Failed to download water bodies: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_combined_workflow():
    """Test complete workflow with multiple downloads."""
    print("\n" + "="*60)
    print("TEST 4: Combined Workflow")
    print("="*60)
    
    try:
        download_dir = tempfile.mkdtemp()
        
        # Create InputsManager with country code only
        print("Creating InputsManager with country_iso3='MWI'...")
        inputs = InputsManager(
            country_iso3="MWI",
            admin_level=2,
            auto_download=True,
            download_dir=download_dir,
        )
        
        # Check admin boundaries were auto-downloaded
        assert inputs.has_admin(), "Admin boundaries should be available"
        print(f"✓ Admin boundaries: {len(inputs.get_admin_gdf())} features")
        
        # Optionally download population
        print("\nDownloading population raster...")
        inputs.download_population_raster(year=2020)
        assert inputs.has_population(), "Population raster should be available"
        print("✓ Population raster downloaded")
        
        # Optionally download water
        print("\nDownloading water bodies...")
        inputs.download_water_bodies()
        if inputs.has_water():
            print(f"✓ Water bodies: {len(inputs.get_water_gdf())} features")
        else:
            print("⚠ No water bodies found")
        
        print("\n✓ Complete workflow successful!")
        return True
        
    except Exception as e:
        print(f"✗ Combined workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TESTING DOWNLOAD FUNCTIONALITY")
    print("="*60)
    print("\nNote: These tests require internet connection.")
    print("Downloads may take a few moments...\n")
    
    results = {
        "Admin Boundaries": test_admin_download(),
        "Population Raster": test_population_download(),
        "Water Bodies": test_water_download(),
        "Combined Workflow": test_combined_workflow(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
