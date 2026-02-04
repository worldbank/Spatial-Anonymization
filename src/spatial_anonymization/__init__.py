"""
Spatial Anonymization Library

A toolkit for anonymizing survey GPS coordinates using DHS-style displacement,
donut geomasking, and population-adaptive methods.
"""

__version__ = "1.0.0"

# High-level user-facing API
from spatial_anonymization.api import anonymize_survey_data, anonymize_survey_data_from_config

# Core classes for advanced usage
from spatial_anonymization.anonymizer import SurveyAnonymizer
from spatial_anonymization.config import AnonymizationConfig, DataSourcesConfig
from spatial_anonymization.inputs_manager import InputsManager

# Make commonly used items available at package level
__all__ = [
    # Simple API
    "anonymize_survey_data",
    "anonymize_survey_data_from_config",
    # Advanced classes
    "SurveyAnonymizer",
    "AnonymizationConfig",
    "DataSourcesConfig",
    "InputsManager",
]
