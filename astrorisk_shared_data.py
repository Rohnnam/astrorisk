"""
ASTRORISK SHARED DATA MODULE
This module is used by both the backend (data collection) and frontend (Streamlit)
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Shared data file location
DATA_FILE = Path("astrorisk_live_data.json")

def save_live_data(sitrep_data):
    """
    Save the latest SITREP data to a JSON file.
    Called by the backend after each pipeline run.
    """
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(sitrep_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving live data: {e}")
        return False

def load_live_data():
    """
    Load the latest SITREP data from JSON file.
    Called by the Streamlit frontend for display.
    Returns None if file doesn't exist or is invalid.
    """
    try:
        if DATA_FILE.exists():
            with open(DATA_FILE, 'r') as f:
                data = json.load(f)
            return data
        return None
    except Exception as e:
        print(f"Error loading live data: {e}")
        return None

def get_synthetic_fallback():
    """
    Return synthetic data as fallback when no live data is available.
    """
    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'ml_forecast': {
            'storm_probability': 0.13,
            'detected_pattern': 'Waiting for live data...',
            'confidence': 0.0,
            'urgency_multiplier': 1.0
        },
        'sectors': {
            'satellite': {
                'sector': 'Satellite Operations',
                'risk_score': 30,
                'category': 'green',
                'advisory': 'System initializing - awaiting live telemetry',
                'primary_driver': 'electron_norm'
            },
            'aviation': {
                'sector': 'Aviation & Communications',
                'risk_score': 35,
                'category': 'yellow',
                'advisory': 'System initializing - awaiting live telemetry',
                'primary_driver': 'xray_norm'
            },
            'power_grid': {
                'sector': 'Power Grid Stability',
                'risk_score': 20,
                'category': 'green',
                'advisory': 'System initializing - awaiting live telemetry',
                'primary_driver': 'bz_norm'
            }
        },
        'overall_advisory': 'System initializing - live monitoring will begin shortly',
        'raw_data': {
            'kp': 2.0,
            'xray_flux': 1e-8,
            'proton_flux': 1.0,
            'electron_flux': 100.0,
            'wind_speed': 400.0,
            'density': 5.0,
            'temperature': 1e5,
            'bz_gsm': 0.0
        },
        'normalized_data': {
            'kp_norm': 0.22,
            'xray_norm': 0.10,
            'proton_norm': 0.15,
            'electron_norm': 0.20,
            'wind_speed_norm': 0.14,
            'bz_norm': 0.50,
            'density_norm': 0.35,
            'temperature_norm': 0.17
        }
    }