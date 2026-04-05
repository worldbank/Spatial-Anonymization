"""
Spatial Anonymization Library

A toolkit for anonymizing survey GPS coordinates using DHS-style displacement,
donut geomasking, and population-adaptive methods.
"""

__version__ = "1.0.0"

# High-level user-facing API
from spatial_anonymization.api import (
    anonymize_survey_coordinates,
    export_anonymized_data
)

# Core classes for advanced usage
from spatial_anonymization.core.anonymizer import SurveyAnonymizer
from spatial_anonymization.config.params import AnonymizationParams, DataSourcesConfig
from spatial_anonymization.data.inputs_manager import InputsManager

# Diagnostics and visualization
from spatial_anonymization.diagnostics.visualization import plot_displacement_comparison

# Presets module - import from config submodule
from spatial_anonymization.config import presets

# Make commonly used items available at package level
__all__ = [
    # Simple API
    "anonymize_survey_coordinates",
    "export_anonymized_data",
    # Advanced classes
    "SurveyAnonymizer",
    "AnonymizationParams",
    "DataSourcesConfig",
    "InputsManager",
    # Diagnostics
    "plot_displacement_comparison",
    # Presets module (use: from spatial_anonymization.config.presets import DHS)
    "presets",
]
