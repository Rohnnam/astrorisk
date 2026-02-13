"""
ASTRORISK BACKEND - INTEGRATED WITH FRONTEND
Collects live data and saves it for Streamlit frontend display
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from collections import deque
from typing import Dict
import os
from mistralai import Mistral
import time
from astrorisk_shared_data import save_live_data

# =============================================================================
# STEP 1: DATA INGESTION (Real-time NOAA/NASA data) - ROBUST VERSION
# =============================================================================

class AstroRiskIngestor:
    """Fetches real-time space weather data from NOAA/NASA APIs with robust error handling"""
    def __init__(self):
        self.swpc_url = "https://services.swpc.noaa.gov"
        # Cache for fallback values when API fails
        self.last_valid_data = {
            'kp': 2.0,
            'xray_flux': 1e-8,
            'proton_flux': 1.0,
            'electron_flux': 100.0,
            'wind_speed': 400.0,
            'density': 5.0,
            'temperature': 1e5,
            'bz_gsm': 0.0
        }

    def fetch_kp_index(self):
        import requests
        try:
            url = f"{self.swpc_url}/products/noaa-planetary-k-index.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data[1:], columns=data[0]).tail(1)
            
            # Update cache
            if not df.empty and 'Kp' in df.columns:
                self.last_valid_data['kp'] = float(df['Kp'].values[0])
            return df
        except Exception as e:
            print(f"⚠️  Kp Index fetch error: {e} - Using cached value")
            return pd.DataFrame([{'Kp': self.last_valid_data['kp']}])

    def fetch_xray_flux(self):
        import requests
        try:
            url = f"{self.swpc_url}/json/goes/primary/xrays-1-day.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data).tail(1)
            
            # Update cache
            if not df.empty and 'flux' in df.columns:
                self.last_valid_data['xray_flux'] = float(df['flux'].values[0])
            return df
        except Exception as e:
            print(f"⚠️  X-ray flux fetch error: {e} - Using cached value")
            return pd.DataFrame([{'flux': self.last_valid_data['xray_flux']}])

    def fetch_proton_flux(self):
        import requests
        try:
            url = f"{self.swpc_url}/json/goes/primary/integral-protons-1-day.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data).tail(1)
            
            # Update cache
            if not df.empty and 'flux' in df.columns:
                self.last_valid_data['proton_flux'] = float(df['flux'].values[0])
            return df
        except Exception as e:
            print(f"⚠️  Proton flux fetch error: {e} - Using cached value")
            return pd.DataFrame([{'flux': self.last_valid_data['proton_flux']}])

    def fetch_electron_flux(self):
        import requests
        try:
            url = f"{self.swpc_url}/json/goes/primary/integral-electrons-1-day.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data).tail(1)
            
            # Update cache
            if not df.empty and 'flux' in df.columns:
                self.last_valid_data['electron_flux'] = float(df['flux'].values[0])
            return df
        except Exception as e:
            print(f"⚠️  Electron flux fetch error: {e} - Using cached value")
            return pd.DataFrame([{'flux': self.last_valid_data['electron_flux']}])

    def fetch_solar_wind_omni(self):
        import requests
        mag_url = f"{self.swpc_url}/products/solar-wind/mag-1-day.json"
        plasma_url = f"{self.swpc_url}/products/solar-wind/plasma-1-day.json"
        
        try:
            # Fetch with timeout
            mag_res = requests.get(mag_url, timeout=10)
            plasma_res = requests.get(plasma_url, timeout=10)
            
            mag_res.raise_for_status()
            plasma_res.raise_for_status()
            
            mag_data = mag_res.json()
            plasma_data = plasma_res.json()
            
            # Parse carefully with validation
            mag_df = pd.DataFrame(mag_data[1:], columns=mag_data[0])
            plasma_df = pd.DataFrame(plasma_data[1:], columns=plasma_data[0])
            
            # Get last valid entry
            mag_df = mag_df.tail(1)
            plasma_df = plasma_df.tail(1)
            
            # Validate required columns exist
            required_mag = ['time_tag', 'bz_gsm']
            required_plasma = ['speed', 'density', 'temperature']
            
            if not all(col in mag_df.columns for col in required_mag):
                raise ValueError(f"Missing magnetic field columns. Available: {mag_df.columns.tolist()}")
            
            if not all(col in plasma_df.columns for col in required_plasma):
                raise ValueError(f"Missing plasma columns. Available: {plasma_df.columns.tolist()}")
            
            # Extract and validate data
            omni_lite = {
                "time_tag": mag_df['time_tag'].values[0],
                "wind_speed": float(plasma_df['speed'].values[0]),
                "density": float(plasma_df['density'].values[0]),
                "temperature": float(plasma_df['temperature'].values[0]),
                "bz_gsm": float(mag_df['bz_gsm'].values[0])
            }
            
            # Update cache with valid data
            self.last_valid_data['wind_speed'] = omni_lite['wind_speed']
            self.last_valid_data['density'] = omni_lite['density']
            self.last_valid_data['temperature'] = omni_lite['temperature']
            self.last_valid_data['bz_gsm'] = omni_lite['bz_gsm']
            
            return pd.DataFrame([omni_lite])
            
        except Exception as e:
            print(f"⚠️  Solar Wind Error: {e} - Using cached values")
            # Return cached values as fallback
            omni_lite = {
                "time_tag": datetime.now(timezone.utc).isoformat(),
                "wind_speed": self.last_valid_data['wind_speed'],
                "density": self.last_valid_data['density'],
                "temperature": self.last_valid_data['temperature'],
                "bz_gsm": self.last_valid_data['bz_gsm']
            }
            return pd.DataFrame([omni_lite])

    def get_raw_data_dict(self):
        """Return current raw data values for frontend"""
        return self.last_valid_data.copy()


# =============================================================================
# STEP 2: NORMALIZATION (0-1 scaling)
# =============================================================================

class NormalizationLayer:
    """Normalizes raw space weather parameters to 0-1 scale"""
    def __init__(self):
        self.thresholds = {
            'kp': {'min': 0, 'max': 9},
            'solar_wind_speed': {'min': 300, 'max': 1000},
            'xray_flux': {'min': 1e-9, 'max': 1e-3},
            'proton_flux': {'min': 0.1, 'max': 1e5},
            'electron_flux': {'min': 1, 'max': 1e6},
            'bz_gsm': {'min': -30, 'max': 30},
            'density': {'min': 1, 'max': 100},
            'temperature': {'min': 1e4, 'max': 2e6}
        }
    
    def normalize_kp(self, kp_value):
        return np.clip(kp_value / 9.0, 0, 1)
    
    def normalize_xray_flux(self, flux_value):
        flux_clamped = np.clip(flux_value, 1e-9, 1e-3)
        log_flux = np.log10(flux_clamped)
        return np.clip((log_flux - np.log10(1e-9)) / (np.log10(1e-3) - np.log10(1e-9)), 0, 1)
    
    def normalize_proton_flux(self, flux_value):
        flux_clamped = np.clip(flux_value, 0.1, 1e5)
        log_flux = np.log10(flux_clamped)
        return np.clip((log_flux - np.log10(0.1)) / (np.log10(1e5) - np.log10(0.1)), 0, 1)
    
    def normalize_electron_flux(self, flux_value):
        flux_clamped = np.clip(flux_value, 1, 1e6)
        log_flux = np.log10(flux_clamped)
        return np.clip((log_flux - np.log10(1)) / (np.log10(1e6) - np.log10(1)), 0, 1)
    
    def normalize_solar_wind_speed(self, speed_value):
        return np.clip((speed_value - 300) / 700, 0, 1)
    
    def normalize_bz_gsm(self, bz_value):
        return np.clip((30 - np.clip(bz_value, -30, 30)) / 60, 0, 1)
    
    def normalize_density(self, density_value):
        density_clamped = np.clip(density_value, 1, 100)
        log_density = np.log10(density_clamped)
        return np.clip((log_density - np.log10(1)) / (np.log10(100) - np.log10(1)), 0, 1)
    
    def normalize_temperature(self, temp_value):
        temp_clamped = np.clip(temp_value, 1e4, 2e6)
        log_temp = np.log10(temp_clamped)
        return np.clip((log_temp - np.log10(1e4)) / (np.log10(2e6) - np.log10(1e4)), 0, 1)


# =============================================================================
# STEP 3: HISTORY BUFFER (For temporal features)
# =============================================================================

HISTORY_LENGTH = 72  # 6 hours @ 5-minute cadence
astrorisk_history = deque(maxlen=HISTORY_LENGTH)

def append_to_history(normalized_data: Dict[str, float]) -> pd.DataFrame:
    """Append current snapshot to rolling history"""
    record = {
        'time_tag': datetime.now(timezone.utc),
        'kp_norm': normalized_data.get('kp_norm'),
        'xray_norm': normalized_data.get('xray_norm'),
        'proton_norm': normalized_data.get('proton_norm'),
        'electron_norm': normalized_data.get('electron_norm'),
        'wind_speed_norm': normalized_data.get('wind_speed_norm'),
        'bz_norm': normalized_data.get('bz_norm'),
        'density_norm': normalized_data.get('density_norm'),
        'temperature_norm': normalized_data.get('temperature_norm')
    }
    astrorisk_history.append(record)
    return pd.DataFrame(list(astrorisk_history))


# =============================================================================
# STEP 4: FEATURE EXTRACTION (Temporal intelligence) - NUMPY COMPATIBLE
# =============================================================================

class FeatureExtractor:
    """Extracts temporal features from historical data"""
    def __init__(self):
        self.min_samples = {'bz_sustained': 12, 'proton_baseline': 48}
    
    def extract_features(self, history_df):
        """Extract all temporal features"""
        if history_df.empty:
            return self._empty_features()
        
        features = {
            'bz_sustained_stress': self._calc_bz_sustained(history_df),
            'wind_acceleration': self._calc_wind_accel(history_df),
            'proton_spike_intensity': self._calc_proton_spike(history_df),
            'plasma_pressure': self._calc_plasma_pressure(history_df),
            'kp_persistence': self._calc_kp_persistence(history_df),
            'combined_storm_score': self._calc_combined(history_df),
            'feature_confidence': min(len(history_df) / 48, 1.0),
            'data_freshness': 1.0
        }
        return features
    
    def _calc_bz_sustained(self, df):
        if len(df) < 12:
            return float(df['bz_norm'].iloc[-1]) if 'bz_norm' in df.columns else 0.5
        recent = df.tail(12)
        bz_values = recent['bz_norm'].values
        southward_stress = np.maximum(bz_values - 0.5, 0)
        # Use trapz instead of trapezoid for NumPy compatibility
        stress_integral = np.trapz(southward_stress)
        return min(stress_integral / (0.5 * 12), 1.0)
    
    def _calc_wind_accel(self, df):
        if len(df) < 3:
            return 0.0
        recent = df.tail(4)
        wind_speed = recent['wind_speed_norm'].values
        if len(wind_speed) >= 2:
            derivatives = np.diff(wind_speed)
            positive_accel = np.maximum(derivatives, 0)
            return min(np.mean(positive_accel) / 0.15, 1.0)
        return 0.0
    
    def _calc_proton_spike(self, df):
        if len(df) < 48:
            return float(df['proton_norm'].iloc[-1]) if 'proton_norm' in df.columns else 0.0
        baseline_avg = df.tail(48)['proton_norm'].mean()
        current = df['proton_norm'].iloc[-1]
        spike_ratio = (current - baseline_avg) / (baseline_avg + 0.01)
        return max(min(spike_ratio / 3.0, 1.0), 0.0)
    
    def _calc_plasma_pressure(self, df):
        if df.empty:
            return 0.0
        recent = df.tail(6) if len(df) >= 6 else df
        densities = recent['density_norm'].values
        speeds = recent['wind_speed_norm'].values
        pressures = densities * (speeds ** 2)
        return min(np.mean(pressures), 1.0)
    
    def _calc_kp_persistence(self, df):
        if len(df) < 6:
            return float(df['kp_norm'].iloc[-1]) if 'kp_norm' in df.columns else 0.0
        recent = df.tail(24) if len(df) >= 24 else df
        elevated_mask = recent['kp_norm'].values > 0.33
        if np.any(elevated_mask):
            runs = []
            current_run = 0
            for val in elevated_mask:
                if val:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                runs.append(current_run)
            return min(max(runs) / 24.0, 1.0) if runs else 0.0
        return 0.0
    
    def _calc_combined(self, df):
        if df.empty:
            return 0.0
        current = df.iloc[-1]
        weights = {'kp_norm': 0.25, 'bz_norm': 0.20, 'proton_norm': 0.15, 
                   'electron_norm': 0.10, 'wind_speed_norm': 0.15, 
                   'density_norm': 0.10, 'temperature_norm': 0.05}
        score = sum(float(current[k]) * w for k, w in weights.items() if k in current)
        return min(score / sum(weights.values()), 1.0)
    
    def _empty_features(self):
        return {'bz_sustained_stress': 0.5, 'wind_acceleration': 0.0, 
                'proton_spike_intensity': 0.0, 'plasma_pressure': 0.0,
                'kp_persistence': 0.0, 'combined_storm_score': 0.0,
                'feature_confidence': 0.0, 'data_freshness': 0.0}


# =============================================================================
# STEP 5: ML FORECASTING (Storm probability prediction)
# =============================================================================

class AstroRiskPredictor:
    """ML-based storm probability forecasting"""
    def __init__(self):
        self.feature_weights = {
            'bz_sustained_stress': 0.25,
            'wind_acceleration': 0.20,
            'proton_spike_intensity': 0.15,
            'plasma_pressure': 0.15,
            'kp_persistence': 0.10,
            'combined_storm_score': 0.15
        }
    
    def predict(self, features: Dict, history_df: pd.DataFrame) -> Dict:
        """Predict storm probability"""
        probability = self._forest_surrogate(features)
        
        if len(history_df) < 12:
            return {
                'storm_probability': round(probability, 3),
                'detected_pattern': "Insufficient temporal context (warming up)",
                'confidence': 0.25,
                'urgency_multiplier': 1.00
            }
        
        pattern = self._detect_pattern(features)
        confidence = features.get('feature_confidence', 0.5)
        urgency = self._urgency_multiplier(probability)
        
        return {
            'storm_probability': round(probability, 3),
            'detected_pattern': pattern,
            'confidence': round(confidence, 3),
            'urgency_multiplier': urgency
        }
    
    def _forest_surrogate(self, features):
        weighted_sum = sum(features.get(k, 0.0) * w for k, w in self.feature_weights.items())
        raw_score = weighted_sum / sum(self.feature_weights.values())
        probability = 1 / (1 + np.exp(-8 * (raw_score - 0.5)))
        return float(np.clip(probability, 0.0, 1.0))
    
    def _detect_pattern(self, f):
        if f.get('bz_sustained_stress', 0) > 0.65 and f.get('wind_acceleration', 0) > 0.50:
            return "CME Shock Front + Sustained Southward IMF"
        if f.get('proton_spike_intensity', 0) > 0.45:
            return "Solar Particle Event Precursor"
        if f.get('bz_sustained_stress', 0) > 0.65:
            return "Magnetic Reconnection Dominant"
        return "No Dominant Storm Precursor"
    
    def _urgency_multiplier(self, probability):
        if probability >= 0.85:
            return 1.35
        elif probability >= 0.70:
            return 1.22
        elif probability >= 0.50:
            return 1.10
        return 1.00


# =============================================================================
# STEP 6: RISK SCORE ENGINE (Sector-specific risk assessment)
# =============================================================================

class AstroRiskEngine:
    """Calculates sector-specific risk scores"""
    def __init__(self):
        self.sector_weights = {
            'satellite': {'electron_norm': 0.40, 'proton_norm': 0.30, 'kp_norm': 0.20, 'wind_speed_norm': 0.10},
            'aviation': {'xray_norm': 0.50, 'proton_norm': 0.30, 'kp_norm': 0.20},
            'power_grid': {'kp_norm': 0.50, 'bz_norm': 0.30, 'plasma_pressure': 0.20}
        }
    
    def calculate_risk_scores(self, normalized_data, derived_features=None):
        """Calculate risk scores for all sectors"""
        all_data = normalized_data.copy()
        if derived_features:
            all_data['plasma_pressure'] = derived_features.get('plasma_pressure', 0.0)
        
        results = {
            'satellite': self._calc_satellite_risk(all_data),
            'aviation': self._calc_aviation_risk(all_data),
            'power_grid': self._calc_grid_risk(all_data),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        return results
    
    def _calc_satellite_risk(self, data):
        weights = self.sector_weights['satellite']
        score = sum(data.get(k, 0.0) * w for k, w in weights.items())
        risk_score = int(round(score * 100))
        category = self._categorize(risk_score)
        primary = max(((k, data.get(k, 0)*w) for k, w in weights.items()), key=lambda x: x[1])
        return {
            'sector': 'Satellite Operations',
            'risk_score': risk_score,
            'category': category,
            'primary_driver': {'parameter': primary[0], 'contribution': round(primary[1]*100, 1)},
            'threats': self._satellite_threats(data)
        }
    
    def _calc_aviation_risk(self, data):
        weights = self.sector_weights['aviation']
        score = sum(data.get(k, 0.0) * w for k, w in weights.items())
        risk_score = int(round(score * 100))
        category = self._categorize(risk_score)
        primary = max(((k, data.get(k, 0)*w) for k, w in weights.items()), key=lambda x: x[1])
        return {
            'sector': 'Aviation & Communications',
            'risk_score': risk_score,
            'category': category,
            'primary_driver': {'parameter': primary[0], 'contribution': round(primary[1]*100, 1)},
            'threats': self._aviation_threats(data)
        }
    
    def _calc_grid_risk(self, data):
        weights = self.sector_weights['power_grid']
        score = sum(data.get(k, 0.0) * w for k, w in weights.items())
        risk_score = int(round(score * 100))
        category = self._categorize(risk_score)
        primary = max(((k, data.get(k, 0)*w) for k, w in weights.items()), key=lambda x: x[1])
        return {
            'sector': 'Power Grid Stability',
            'risk_score': risk_score,
            'category': category,
            'primary_driver': {'parameter': primary[0], 'contribution': round(primary[1]*100, 1)},
            'threats': self._grid_threats(data)
        }
    
    def _categorize(self, score):
        if score <= 33:
            return 'green'
        elif score <= 66:
            return 'yellow'
        return 'red'
    
    def _satellite_threats(self, data):
        threats = []
        if data.get('electron_norm', 0) > 0.6:
            threats.append("Deep dielectric charging risk")
        if data.get('proton_norm', 0) > 0.5:
            threats.append("Single event upset (SEU) risk")
        if data.get('kp_norm', 0) > 0.6:
            threats.append("Increased atmospheric drag")
        return threats if threats else ["Nominal conditions"]
    
    def _aviation_threats(self, data):
        threats = []
        if data.get('xray_norm', 0) > 0.6:
            threats.append("HF radio blackout")
        if data.get('proton_norm', 0) > 0.5:
            threats.append("Elevated radiation exposure")
        if data.get('kp_norm', 0) > 0.5:
            threats.append("GPS navigation degradation")
        return threats if threats else ["Nominal conditions"]
    
    def _grid_threats(self, data):
        threats = []
        if data.get('kp_norm', 0) > 0.6:
            threats.append("Geomagnetically Induced Currents (GICs)")
        if data.get('bz_norm', 0) > 0.6:
            threats.append("Magnetic reconnection (southward Bz)")
        if data.get('plasma_pressure', 0) > 0.5:
            threats.append("Magnetosphere compression")
        return threats if threats else ["Nominal conditions"]


# =============================================================================
# STEP 7: LLM ADVISORY LAYER (Human interface) - MISTRAL AI VERSION
# =============================================================================

def get_llm_advisory(sector, risk_score, driver, storm_prob):
    """Generates an operator advisory using Mistral AI."""
    MISTRAL_API_KEY = "pn2dxv5anni4FPKtlkFyBlpdClBjRrQ2" 
    
    if not MISTRAL_API_KEY or "YOUR_MISTRAL" in MISTRAL_API_KEY:
        return f"DATA_ONLY: {sector} Risk {risk_score}/100 (Drivers: {driver}). Monitoring active."

    try:
        client = Mistral(api_key=MISTRAL_API_KEY)

        prompt = f"""You are ASTRORISK, a Space Weather Operations AI.
Current Status:
- Sector: {sector}
- Risk Score: {risk_score}/100
- Main Driver: {driver}
- Storm Probability: {storm_prob}%

Task: Write a ONE-SENTENCE military-style situation update (SITREP) for the operator.
Format: "STATUS: [Nominal/Caution/Critical]: [Actionable advice]."
Keep it under 20 words."""

        chat_response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
        )

        return chat_response.choices[0].message.content

    except Exception as e:
        print(f"⚠️ Mistral API Error: {e}")
        return f"FALLBACK: {sector} Risk {risk_score}/100. Check telemetry manually."


class AstroRiskAdvisor:
    """LLM-powered advisory translation layer - Mistral AI Edition"""
    def __init__(self, mistral_api_key):
        self.api_key = mistral_api_key
    
    def generate_situation_report(self, all_sectors):
        """Generate complete SITREP with LLM advisories"""
        sitrep = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'sectors': {},
            'ml_forecast': all_sectors.get('ml_forecast', {}),
            'overall_advisory': None
        }
        
        for sector_key in ['satellite', 'aviation', 'power_grid']:
            if sector_key not in all_sectors:
                continue
            
            sector_data = all_sectors[sector_key]
            drivers = self._extract_drivers(sector_data)
            driver_text = ', '.join(drivers[:2]) if drivers else 'General space weather'
            storm_prob = all_sectors.get('ml_forecast', {}).get('storm_probability', 0.0)
            
            advisory = get_llm_advisory(
                sector=sector_data['sector'],
                risk_score=sector_data['risk_score'],
                driver=driver_text,
                storm_prob=int(storm_prob * 100)
            )
            
            sitrep['sectors'][sector_key] = {
                'sector': sector_data['sector'],
                'risk_score': sector_data['risk_score'],
                'category': sector_data['category'],
                'advisory': advisory,
                'primary_driver': sector_data['primary_driver']['parameter']
            }
        
        sitrep['overall_advisory'] = self._overall_advisory(all_sectors)
        return sitrep
    
    def _extract_drivers(self, sector_data):
        """Convert technical parameters to human-readable"""
        driver_map = {
            'electron_norm': 'High Electron Flux',
            'proton_norm': 'High Proton Flux',
            'kp_norm': 'Elevated Kp Index',
            'xray_norm': 'Solar Flare Activity',
            'wind_speed_norm': 'Fast Solar Wind',
            'bz_norm': 'Southward IMF Bz',
            'plasma_pressure': 'Plasma Pressure Surge'
        }
        
        primary = sector_data.get('primary_driver', {}).get('parameter', '')
        drivers = [driver_map.get(primary, 'Space weather activity')]
        
        threats = sector_data.get('threats', [])
        if threats and threats[0] != 'Nominal conditions' and len(drivers) < 2:
            drivers.append(threats[0])
        
        return drivers
    
    def _overall_advisory(self, all_sectors):
        """Generate overall system advisory"""
        scores = [all_sectors[k]['risk_score'] for k in ['satellite', 'aviation', 'power_grid'] if k in all_sectors]
        red_count = sum(1 for s in scores if s >= 67)
        yellow_count = sum(1 for s in scores if 34 <= s < 67)
        
        if red_count > 0:
            return f"SYSTEM ALERT: {red_count} sector(s) critical — coordinated response recommended."
        elif yellow_count >= 2:
            return f"HEIGHTENED WATCH: {yellow_count} sectors elevated — monitor closely."
        elif yellow_count == 1:
            return "ADVISORY: Single-sector caution — standard monitoring active."
        return "ALL CLEAR: All sectors nominal."


# =============================================================================
# 🎯 HACKATHON WARM-UP: SYNTHETIC DATA INJECTION
# =============================================================================

def warmup_with_synthetic_data():
    """
    Pre-fill history buffer with synthetic data for immediate pattern detection.
    This simulates the system having been running for hours.
    
    🚨 HACKATHON MODE: This lets you demo the ML patterns without waiting 1+ hour!
    """
    print("\n🔥 HACKATHON MODE: Warming up with synthetic history...")
    print("   Simulating 50+ samples of space weather data for instant ML pattern detection")
    
    # Scenario 1: Build up to a moderate storm (samples 0-30)
    for i in range(30):
        progression = i / 30.0  # 0.0 to 1.0
        
        fake_data = {
            'kp_norm': 0.1 + (progression * 0.4),  # Kp rises from 1 to 4
            'bz_norm': 0.3 + (progression * 0.4),  # Bz becomes more southward
            'wind_speed_norm': 0.2 + (progression * 0.3),  # Wind accelerates
            'xray_norm': 0.1 + (progression * 0.2),  # Solar activity increases
            'proton_norm': 0.05 + (progression * 0.15),  # Proton flux rises
            'electron_norm': 0.1 + (progression * 0.3),  # Electron flux increases
            'density_norm': 0.1 + (progression * 0.2),
            'temperature_norm': 0.15 + (progression * 0.25)
        }
        append_to_history(fake_data)
    
    # Scenario 2: Sustained elevated conditions (samples 31-50)
    for i in range(20):
        # Add some variability
        noise = np.random.uniform(-0.05, 0.05)
        
        fake_data = {
            'kp_norm': 0.5 + noise,  # Kp ~ 4-5 (moderate storm)
            'bz_norm': 0.7 + noise,  # Strongly southward
            'wind_speed_norm': 0.5 + noise,  # Fast wind
            'xray_norm': 0.3 + noise,  # M-class flare range
            'proton_norm': 0.2 + noise,  # Elevated protons
            'electron_norm': 0.4 + noise,  # High electrons
            'density_norm': 0.3 + noise,
            'temperature_norm': 0.4 + noise
        }
        append_to_history(fake_data)
    
    print(f"✓ Synthetic warm-up complete: {len(astrorisk_history)} samples in buffer")
    print(f"   ML patterns should now be IMMEDIATELY detectable!\n")


# =============================================================================
# MASTER EXECUTION PIPELINE WITH FRONTEND INTEGRATION
# =============================================================================

def run_astrorisk_pipeline(mistral_api_key):
    """Complete AstroRisk pipeline execution with robust error handling"""
    
    print("="*80)
    print("ASTRORISK COMPLETE PIPELINE - MISTRAL AI EDITION")
    print("End-to-End Execution")
    print("="*80)
    
    # STEP 1: Fetch real-time data (with error handling)
    print("\n[1/7] Fetching real-time space weather data...")
    ingestor = AstroRiskIngestor()
    
    try:
        kp_df = ingestor.fetch_kp_index()
        xray_df = ingestor.fetch_xray_flux()
        proton_df = ingestor.fetch_proton_flux()
        electron_df = ingestor.fetch_electron_flux()
        solar_wind_df = ingestor.fetch_solar_wind_omni()
        print("✓ Data ingestion complete")
    except Exception as e:
        print(f"✗ Critical data ingestion error: {e}")
        return None
    
    # STEP 2: Normalize data (with safe extraction)
    print("\n[2/7] Normalizing parameters to 0-1 scale...")
    normalizer = NormalizationLayer()
    
    try:
        normalized_data = {
            'kp_norm': normalizer.normalize_kp(float(kp_df['Kp'].values[0])),
            'xray_norm': normalizer.normalize_xray_flux(float(xray_df['flux'].values[0])),
            'proton_norm': normalizer.normalize_proton_flux(float(proton_df['flux'].values[0])),
            'electron_norm': normalizer.normalize_electron_flux(float(electron_df['flux'].values[0])),
            'wind_speed_norm': normalizer.normalize_solar_wind_speed(float(solar_wind_df['wind_speed'].values[0])),
            'bz_norm': normalizer.normalize_bz_gsm(float(solar_wind_df['bz_gsm'].values[0])),
            'density_norm': normalizer.normalize_density(float(solar_wind_df['density'].values[0])),
            'temperature_norm': normalizer.normalize_temperature(float(solar_wind_df['temperature'].values[0]))
        }
        print("✓ Normalization complete")
    except Exception as e:
        print(f"✗ Normalization error: {e}")
        return None
    
    # STEP 3: Append to history
    print("\n[3/7] Updating temporal history buffer...")
    history_df = append_to_history(normalized_data)
    print(f"✓ History buffer: {len(history_df)} samples")
    
    # STEP 4: Extract temporal features
    print("\n[4/7] Extracting temporal features...")
    extractor = FeatureExtractor()
    derived_features = extractor.extract_features(history_df)
    print(f"✓ Features extracted (confidence: {derived_features['feature_confidence']:.2f})")
    
    # STEP 5: ML forecast
    print("\n[5/7] Running ML storm probability forecast...")
    predictor = AstroRiskPredictor()
    ml_output = predictor.predict(derived_features, history_df)
    print(f"✓ Storm probability: {ml_output['storm_probability']*100:.1f}%")
    print(f"  Pattern: {ml_output['detected_pattern']}")
    
    # STEP 6: Calculate risk scores
    print("\n[6/7] Calculating sector-specific risk scores...")
    engine = AstroRiskEngine()
    risk_scores = engine.calculate_risk_scores(normalized_data, derived_features)
    risk_scores['ml_forecast'] = ml_output
    print("✓ Risk scores calculated")
    
    # STEP 7: Generate LLM advisories using Mistral AI
    print("\n[7/7] Generating operator advisories with Mistral AI...")
    advisor = AstroRiskAdvisor(mistral_api_key)
    sitrep = advisor.generate_situation_report(risk_scores)
    print("✓ SITREP generation complete")
    
    # 🌟 NEW: Add raw data and normalized data to SITREP for frontend
    sitrep['raw_data'] = ingestor.get_raw_data_dict()
    sitrep['normalized_data'] = normalized_data
    sitrep['temporal_features'] = derived_features
    
    # 🌟 NEW: Save SITREP to shared data file for frontend
    save_live_data(sitrep)
    print("✓ Data saved for frontend display")
    
    # Display final output
    print("\n" + "="*80)
    print("OPERATOR SITUATION REPORT (SITREP)")
    print("="*80)
    print(f"Generated: {sitrep['timestamp']}")
    
    # ML Forecast
    print("\n🔮 ML FORECAST (Next 1-6 Hours):")
    print("-" * 80)
    ml = sitrep['ml_forecast']
    print(f"  Storm Probability (G3+): {ml['storm_probability']*100:.1f}%")
    print(f"  Detected Pattern: {ml['detected_pattern']}")
    print(f"  Confidence: {ml['confidence']:.2f}")
    print(f"  Urgency Multiplier: {ml['urgency_multiplier']:.2f}x")
    
    # Sector advisories
    print("\n📡 SECTOR ADVISORIES (Powered by Mistral AI):")
    print("-" * 80)
    
    category_icons = {'green': '✅', 'yellow': '⚠️', 'red': '🚨'}
    for sector_key, sector_data in sitrep['sectors'].items():
        icon = category_icons.get(sector_data['category'], '❓')
        print(f"\n{icon} {sector_data['sector'].upper()}")
        print(f"   Risk Score: {sector_data['risk_score']}/100")
        print(f"   Primary Driver: {sector_data['primary_driver']}")
        print(f"   Advisory: {sector_data['advisory']}")
    
    # Overall advisory
    print("\n" + "="*80)
    print("OVERALL SITUATION:")
    print("-" * 80)
    print(f"   {sitrep['overall_advisory']}")
    print("="*80)
    
    return sitrep


# =============================================================================
# MAIN EXECUTION WITH SYNTHETIC WARM-UP AND CONTINUOUS MONITORING
# =============================================================================

if __name__ == "__main__":
    MISTRAL_API_KEY = "pn2dxv5anni4FPKtlkFyBlpdClBjRrQ2"
    
    print("🚀 STARTING ASTRORISK LIVE MONITORING...")
    print("   System will continue monitoring even if temporary API failures occur.")
    print("   Frontend will auto-refresh to display live data.\n")
    
    # 🎯 HACKATHON WARM-UP: Pre-fill with synthetic data
    warmup_with_synthetic_data()
    
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    while True:
        try:
            sitrep = run_astrorisk_pipeline(MISTRAL_API_KEY)
            
            if sitrep is not None:
                consecutive_failures = 0  # Reset on success
            else:
                consecutive_failures += 1
                print(f"\n⚠️  Pipeline returned None (Failure {consecutive_failures}/{max_consecutive_failures})")
                
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\n🛑 Too many consecutive failures. Stopping monitoring.")
                    break
            
            # Wait for 5 minutes before the next update
            print("\n⏳ Sleeping for 300 seconds (5 minutes)...")
            time.sleep(300)
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user.")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error in main loop: {e}")
            consecutive_failures += 1
            
            if consecutive_failures >= max_consecutive_failures:
                print(f"\n🛑 Too many consecutive failures. Stopping monitoring.")
                break
            
            print(f"   Retrying in 60 seconds... (Failure {consecutive_failures}/{max_consecutive_failures})")
            time.sleep(60)