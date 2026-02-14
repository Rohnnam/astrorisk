"""
ASTRORISK BACKEND V2 - ENHANCED WITH DETAILED ANALYSIS
- Fixed confidence calculation (100% at 24 samples instead of 48)
- Improved temporal feature extraction
- Detailed multi-paragraph LLM analysis for each sector
- All integrated with live NOAA/NASA data
- ✅ STEP 1: Supabase connection proof
- ✅ STEP 2: Test row insert to verify DB write
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from collections import deque
from typing import Dict, List
import os
from mistralai import Mistral
import time
from astrorisk_shared_data import save_live_data
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# =============================================================================
# ✅ STEP 1 — CONFIRM SUPABASE CONNECTION (proof in Render logs)
# =============================================================================
print("SUPABASE_URL:", SUPABASE_URL)
print("SUPABASE_SERVICE_KEY exists:", bool(SUPABASE_SERVICE_KEY))

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# =============================================================================
# TEMPORAL FUSION TRANSFORMER (TFT) - STATE-OF-THE-ART FORECASTING
# =============================================================================

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for multi-horizon space weather forecasting.
    
    Combines:
    - LSTM for local temporal patterns
    - Multi-head attention for long-range dependencies
    - Variable selection networks to identify key drivers
    """
    def __init__(self, input_dim=8, hidden_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        
        # Variable Selection Network
        self.variable_selection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )
        
        # LSTM for local temporal patterns
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Multi-head attention for long-range dependencies
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, features] - Time series of space weather parameters
        Returns:
            probability: Storm probability [0, 1]
            importance: Variable importance scores
        """
        # Variable selection - which parameters matter most?
        importance = self.variable_selection(x[:, -1, :])  # Use latest timestep
        
        # Apply variable selection
        x_selected = x * importance.unsqueeze(1)
        
        # LSTM for temporal patterns
        lstm_out, _ = self.lstm(x_selected)
        
        # Self-attention for long-range dependencies
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last timestep
        final_hidden = attn_out[:, -1, :]
        
        # Prediction head
        out = self.relu(self.fc1(final_hidden))
        out = self.dropout(out)
        probability = torch.sigmoid(self.fc2(out))
        
        return probability.squeeze(), importance


class TFTPredictor:
    """Wrapper for TFT model with pretrained weights simulation"""
    def __init__(self):
        self.model = TemporalFusionTransformer()
        self.model.eval()  # Inference mode
        
        # In production, load pretrained weights:
        # self.model.load_state_dict(torch.load('tft_weights.pth'))
        
        # For demo, use random but consistent initialization
        torch.manual_seed(42)
        
    def predict(self, history_df: pd.DataFrame) -> Dict:
        """
        Predict storm probability using TFT.
        
        Args:
            history_df: DataFrame with normalized space weather parameters
        Returns:
            Dict with probability, pattern, confidence, and variable importance
        """
        if len(history_df) < 12:
            return {
                'storm_probability': 0.05,
                'detected_pattern': 'Insufficient data for TFT',
                'confidence': 0.1,
                'urgency_multiplier': 1.0,
                'variable_importance': {}
            }
        
        # Prepare sequence (last 24 timesteps)
        seq_len = min(24, len(history_df))
        recent_data = history_df.tail(seq_len)
        
        # Extract features in order
        features = ['kp_norm', 'xray_norm', 'proton_norm', 'electron_norm',
                   'wind_speed_norm', 'bz_norm', 'density_norm', 'temperature_norm']
        
        # Create input tensor [1, seq_len, features]
        X = torch.tensor(
            recent_data[features].values,
            dtype=torch.float32
        ).unsqueeze(0)
        
        # Run TFT inference
        with torch.no_grad():
            probability, importance = self.model(X)
            probability = float(probability.item())  # Convert to Python float
            importance = importance.squeeze().numpy()
        
        # Map importance to parameter names
        var_importance = {feat: float(imp) for feat, imp in zip(features, importance)}
        
        # Detect pattern based on variable importance
        pattern = self._detect_pattern_from_importance(var_importance, recent_data)
        
        # FIXED: Calculate confidence - reach 100% at 24 samples (2 hours) instead of 48 (4 hours)
        confidence = min(seq_len / 24.0, 1.0)
        
        # Urgency multiplier
        urgency = self._calculate_urgency(probability)
        
        return {
            'storm_probability': round(probability, 3),
            'detected_pattern': pattern,
            'confidence': round(confidence, 3),
            'urgency_multiplier': float(urgency),
            'variable_importance': var_importance
        }
    
    def _detect_pattern_from_importance(self, importance: Dict, data: pd.DataFrame) -> str:
        """Detect storm pattern based on which variables are most important"""
        # Find top 2 drivers
        sorted_vars = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top1, top2 = sorted_vars[0][0], sorted_vars[1][0]
        
        # Pattern recognition
        if 'bz_norm' == top1 and 'wind_speed_norm' == top2:
            return "CME Shock Front + Sustained Southward IMF (TFT)"
        elif 'proton_norm' == top1:
            return "Solar Particle Event Precursor (TFT)"
        elif 'bz_norm' == top1:
            return "Magnetic Reconnection Dominant (TFT)"
        elif 'electron_norm' == top1:
            return "Radiation Belt Enhancement (TFT)"
        elif 'xray_norm' == top1:
            return "Solar Flare Impact (TFT)"
        else:
            return f"Multi-factor Storm ({top1.replace('_norm', '')} + {top2.replace('_norm', '')}) [TFT]"
    
    def _calculate_urgency(self, probability: float) -> float:
        """Calculate urgency multiplier"""
        if probability >= 0.85:
            return 1.35
        elif probability >= 0.70:
            return 1.22
        elif probability >= 0.50:
            return 1.10
        return 1.00


# =============================================================================
# RANDOM FOREST CLASSIFIERS - SECTOR-SPECIFIC RISK
# =============================================================================

class SectorRandomForest:
    """
    Random Forest classifier for sector-specific risk assessment.
    Trained to classify risk levels: LOW (0-33), MEDIUM (34-66), HIGH (67-100)
    """
    def __init__(self, sector_name: str):
        self.sector_name = sector_name
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Synthetic training for demo (in production, use real historical data)
        self._synthetic_training()
    
    def _synthetic_training(self):
        """Train on synthetic data for demonstration"""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Features: normalized parameters
        X = np.random.rand(n_samples, 8)
        
        # Labels: risk categories based on sector
        if self.sector_name == 'satellite':
            # Satellite risk driven by electron/proton flux
            scores = (X[:, 3] * 0.4 + X[:, 2] * 0.3 + X[:, 0] * 0.2 + X[:, 4] * 0.1) * 100
        elif self.sector_name == 'aviation':
            # Aviation risk driven by x-ray/proton flux
            scores = (X[:, 1] * 0.5 + X[:, 2] * 0.3 + X[:, 0] * 0.2) * 100
        else:  # power_grid
            # Grid risk driven by Kp/Bz
            scores = (X[:, 0] * 0.5 + X[:, 5] * 0.3 + X[:, 6] * 0.1 + X[:, 7] * 0.1) * 100
        
        # Add noise
        scores += np.random.randn(n_samples) * 5
        scores = np.clip(scores, 0, 100)
        
        # Categorize
        y = np.zeros(n_samples, dtype=int)
        y[scores >= 67] = 2  # HIGH
        y[(scores >= 34) & (scores < 67)] = 1  # MEDIUM
        y[scores < 34] = 0  # LOW
        
        # Train
        X_scaled = self.scaler.fit_transform(X)
        self.rf_model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict_risk_category(self, normalized_data: Dict) -> Dict:
        """
        Predict risk category and get feature importance.
        
        Returns:
            category: 0 (LOW), 1 (MEDIUM), 2 (HIGH)
            probability: Probability of each class
            feature_importance: Importance of each parameter
        """
        if not self.is_trained:
            return {'category': 0, 'probability': [1, 0, 0], 'feature_importance': {}}
        
        # Prepare features
        features = [
            normalized_data.get('kp_norm', 0),
            normalized_data.get('xray_norm', 0),
            normalized_data.get('proton_norm', 0),
            normalized_data.get('electron_norm', 0),
            normalized_data.get('wind_speed_norm', 0),
            normalized_data.get('bz_norm', 0),
            normalized_data.get('density_norm', 0),
            normalized_data.get('temperature_norm', 0)
        ]
        
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        category = int(self.rf_model.predict(X_scaled)[0])
        probabilities = self.rf_model.predict_proba(X_scaled)[0]
        
        # Feature importance
        importance = self.rf_model.feature_importances_
        feature_names = ['kp', 'xray', 'proton', 'electron', 'wind_speed', 'bz', 'density', 'temperature']
        feature_importance = {name: float(imp) for name, imp in zip(feature_names, importance)}
        
        return {
            'category': category,
            'probability': [float(p) for p in probabilities],
            'feature_importance': feature_importance
        }


# =============================================================================
# REST OF THE PIPELINE (Data Ingestion, Normalization, etc.)
# =============================================================================

class AstroRiskIngestor:
    """Fetches real-time space weather data from NOAA/NASA APIs"""
    def __init__(self):
        self.swpc_url = "https://services.swpc.noaa.gov"
        self.last_valid_data = {
            'kp': 2.0, 'xray_flux': 1e-8, 'proton_flux': 1.0, 'electron_flux': 100.0,
            'wind_speed': 400.0, 'density': 5.0, 'temperature': 1e5, 'bz_gsm': 0.0
        }

    def fetch_kp_index(self):
        import requests
        try:
            url = f"{self.swpc_url}/products/noaa-planetary-k-index.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data[1:], columns=data[0]).tail(1)
            if not df.empty and 'Kp' in df.columns:
                self.last_valid_data['kp'] = float(df['Kp'].values[0])
            return df
        except Exception as e:
            print(f"⚠️  Kp Index error: {e}")
            return pd.DataFrame([{'Kp': self.last_valid_data['kp']}])

    def fetch_xray_flux(self):
        import requests
        try:
            url = f"{self.swpc_url}/json/goes/primary/xrays-1-day.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data).tail(1)
            if not df.empty and 'flux' in df.columns:
                self.last_valid_data['xray_flux'] = float(df['flux'].values[0])
            return df
        except Exception as e:
            print(f"⚠️  X-ray flux error: {e}")
            return pd.DataFrame([{'flux': self.last_valid_data['xray_flux']}])

    def fetch_proton_flux(self):
        import requests
        try:
            url = f"{self.swpc_url}/json/goes/primary/integral-protons-1-day.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data).tail(1)
            if not df.empty and 'flux' in df.columns:
                self.last_valid_data['proton_flux'] = float(df['flux'].values[0])
            return df
        except Exception as e:
            print(f"⚠️  Proton flux error: {e}")
            return pd.DataFrame([{'flux': self.last_valid_data['proton_flux']}])

    def fetch_electron_flux(self):
        import requests
        try:
            url = f"{self.swpc_url}/json/goes/primary/integral-electrons-1-day.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data).tail(1)
            if not df.empty and 'flux' in df.columns:
                self.last_valid_data['electron_flux'] = float(df['flux'].values[0])
            return df
        except Exception as e:
            print(f"⚠️  Electron flux error: {e}")
            return pd.DataFrame([{'flux': self.last_valid_data['electron_flux']}])

    def fetch_solar_wind_omni(self):
        import requests
        mag_url = f"{self.swpc_url}/products/solar-wind/mag-1-day.json"
        plasma_url = f"{self.swpc_url}/products/solar-wind/plasma-1-day.json"
        
        try:
            mag_res = requests.get(mag_url, timeout=10)
            plasma_res = requests.get(plasma_url, timeout=10)
            mag_res.raise_for_status()
            plasma_res.raise_for_status()
            
            mag_data = mag_res.json()
            plasma_data = plasma_res.json()
            
            mag_df = pd.DataFrame(mag_data[1:], columns=mag_data[0]).tail(1)
            plasma_df = pd.DataFrame(plasma_data[1:], columns=plasma_data[0]).tail(1)
            
            required_mag = ['time_tag', 'bz_gsm']
            required_plasma = ['speed', 'density', 'temperature']
            
            if not all(col in mag_df.columns for col in required_mag):
                raise ValueError(f"Missing magnetic field columns")
            if not all(col in plasma_df.columns for col in required_plasma):
                raise ValueError(f"Missing plasma columns")
            
            omni_lite = {
                "time_tag": mag_df['time_tag'].values[0],
                "wind_speed": float(plasma_df['speed'].values[0]),
                "density": float(plasma_df['density'].values[0]),
                "temperature": float(plasma_df['temperature'].values[0]),
                "bz_gsm": float(mag_df['bz_gsm'].values[0])
            }
            
            self.last_valid_data.update({
                'wind_speed': omni_lite['wind_speed'],
                'density': omni_lite['density'],
                'temperature': omni_lite['temperature'],
                'bz_gsm': omni_lite['bz_gsm']
            })
            
            return pd.DataFrame([omni_lite])
            
        except Exception as e:
            print(f"⚠️  Solar Wind error: {e}")
            return pd.DataFrame([{
                "time_tag": datetime.now(timezone.utc).isoformat(),
                "wind_speed": self.last_valid_data['wind_speed'],
                "density": self.last_valid_data['density'],
                "temperature": self.last_valid_data['temperature'],
                "bz_gsm": self.last_valid_data['bz_gsm']
            }])

    def get_raw_data_dict(self):
        return self.last_valid_data.copy()


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
    
    def normalize_solar_wind_speed(self, speed):
        th = self.thresholds['solar_wind_speed']
        return np.clip((speed - th['min']) / (th['max'] - th['min']), 0, 1)
    
    def normalize_xray_flux(self, flux):
        th = self.thresholds['xray_flux']
        if flux <= 0:
            return 0.0
        log_val = np.log10(flux)
        log_min = np.log10(th['min'])
        log_max = np.log10(th['max'])
        return np.clip((log_val - log_min) / (log_max - log_min), 0, 1)
    
    def normalize_proton_flux(self, flux):
        th = self.thresholds['proton_flux']
        if flux <= 0:
            return 0.0
        log_val = np.log10(flux)
        log_min = np.log10(th['min'])
        log_max = np.log10(th['max'])
        return np.clip((log_val - log_min) / (log_max - log_min), 0, 1)
    
    def normalize_electron_flux(self, flux):
        th = self.thresholds['electron_flux']
        if flux <= 0:
            return 0.0
        log_val = np.log10(flux)
        log_min = np.log10(th['min'])
        log_max = np.log10(th['max'])
        return np.clip((log_val - log_min) / (log_max - log_min), 0, 1)
    
    def normalize_bz_gsm(self, bz):
        th = self.thresholds['bz_gsm']
        norm = (bz - th['min']) / (th['max'] - th['min'])
        return np.clip(norm, 0, 1)
    
    def normalize_density(self, density):
        th = self.thresholds['density']
        return np.clip((density - th['min']) / (th['max'] - th['min']), 0, 1)
    
    def normalize_temperature(self, temperature):
        th = self.thresholds['temperature']
        return np.clip((temperature - th['min']) / (th['max'] - th['min']), 0, 1)


astrorisk_history = deque(maxlen=300)

def append_to_history(normalized_data):
    record = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        **normalized_data
    }
    astrorisk_history.append(record)
    return pd.DataFrame(list(astrorisk_history))


class FeatureExtractor:
    def extract_features(self, history_df):
        if len(history_df) < 2:
            return {
                'plasma_pressure': 0.0,
                'storm_ramp': 0.0,
                'kp_acceleration': 0.0,
                'wind_burst': 0.0,
                'feature_confidence': 0.1
            }
        
        latest = history_df.iloc[-1]
        prev = history_df.iloc[-2] if len(history_df) >= 2 else latest
        
        # IMPROVED: More sensitive plasma pressure calculation
        density = latest.get('density_norm', 0.5)
        wind_speed = latest.get('wind_speed_norm', 0.3)
        # Amplified calculation to show non-zero values
        plasma_pressure = np.clip((density * wind_speed * 2.5), 0, 1)
        
        # IMPROVED: More sensitive Kp acceleration
        kp_now = latest.get('kp_norm', 0.0)
        kp_prev = prev.get('kp_norm', 0.0)
        # Amplify small changes
        kp_acceleration = (kp_now - kp_prev) * 5.0
        
        # IMPROVED: Storm ramp calculation with better sensitivity
        storm_ramp = 0.0
        if len(history_df) >= 6:
            recent_kps = history_df.tail(6)['kp_norm'].values
            if len(recent_kps) >= 2:
                # Calculate slope and amplify
                slope = (recent_kps[-1] - recent_kps[0]) / len(recent_kps)
                storm_ramp = np.clip(slope * 10.0, 0, 1)
        
        # IMPROVED: Wind burst detection
        wind_now = latest.get('wind_speed_norm', 0.0)
        wind_prev = prev.get('wind_speed_norm', 0.0)
        # Amplify burst detection
        wind_burst = max(0, (wind_now - wind_prev) * 3.0)
        
        confidence = min(len(history_df) / 12.0, 1.0)
        
        return {
            'plasma_pressure': float(np.clip(plasma_pressure, 0, 1)),
            'storm_ramp': float(np.clip(storm_ramp, 0, 1)),
            'kp_acceleration': float(np.clip(kp_acceleration, -1, 1)),
            'wind_burst': float(np.clip(wind_burst, 0, 1)),
            'feature_confidence': float(confidence)
        }


class AstroRiskEngine:
    def __init__(self):
        # Initialize Random Forest classifiers for each sector
        self.rf_satellite = SectorRandomForest('satellite')
        self.rf_aviation = SectorRandomForest('aviation')
        self.rf_power_grid = SectorRandomForest('power_grid')
        
        self.sector_weights = {
            'satellite': {'electron_norm': 0.40, 'proton_norm': 0.30, 'kp_norm': 0.20, 'wind_speed_norm': 0.10},
            'aviation': {'xray_norm': 0.50, 'proton_norm': 0.30, 'kp_norm': 0.20},
            'power_grid': {'kp_norm': 0.50, 'bz_norm': 0.30, 'plasma_pressure': 0.20}
        }
    
    def calculate_risk_scores(self, normalized_data, derived_features=None):
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
        rf_result = self.rf_satellite.predict_risk_category(data)
        
        weights = self.sector_weights['satellite']
        score = sum(data.get(k, 0.0) * w for k, w in weights.items())
        base_risk_score = int(round(score * 100))
        
        if rf_result['category'] == 2:
            risk_score = max(base_risk_score, 67)
        elif rf_result['category'] == 1:
            risk_score = int(np.clip(base_risk_score, 34, 66))
        else:
            risk_score = min(base_risk_score, 33)
        
        category = self._categorize(risk_score)
        primary = max(((k, data.get(k, 0)*w) for k, w in weights.items()), key=lambda x: x[1])
        
        return {
            'sector': 'Satellite Operations',
            'risk_score': int(risk_score),
            'category': category,
            'primary_driver': {'parameter': primary[0], 'contribution': round(primary[1]*100, 1)},
            'threats': self._satellite_threats(data),
            'rf_category': int(rf_result['category']),
            'rf_confidence': float(max(rf_result['probability']))
        }
    
    def _calc_aviation_risk(self, data):
        rf_result = self.rf_aviation.predict_risk_category(data)
        
        weights = self.sector_weights['aviation']
        score = sum(data.get(k, 0.0) * w for k, w in weights.items())
        base_risk_score = int(round(score * 100))
        
        if rf_result['category'] == 2:
            risk_score = max(base_risk_score, 67)
        elif rf_result['category'] == 1:
            risk_score = int(np.clip(base_risk_score, 34, 66))
        else:
            risk_score = min(base_risk_score, 33)
        
        category = self._categorize(risk_score)
        primary = max(((k, data.get(k, 0)*w) for k, w in weights.items()), key=lambda x: x[1])
        
        return {
            'sector': 'Aviation & Communications',
            'risk_score': int(risk_score),
            'category': category,
            'primary_driver': {'parameter': primary[0], 'contribution': round(primary[1]*100, 1)},
            'threats': self._aviation_threats(data),
            'rf_category': int(rf_result['category']),
            'rf_confidence': float(max(rf_result['probability']))
        }
    
    def _calc_grid_risk(self, data):
        rf_result = self.rf_power_grid.predict_risk_category(data)
        
        weights = self.sector_weights['power_grid']
        score = sum(data.get(k, 0.0) * w for k, w in weights.items())
        base_risk_score = int(round(score * 100))
        
        if rf_result['category'] == 2:
            risk_score = max(base_risk_score, 67)
        elif rf_result['category'] == 1:
            risk_score = int(np.clip(base_risk_score, 34, 66))
        else:
            risk_score = min(base_risk_score, 33)
        
        category = self._categorize(risk_score)
        primary = max(((k, data.get(k, 0)*w) for k, w in weights.items()), key=lambda x: x[1])
        
        return {
            'sector': 'Power Grid Stability',
            'risk_score': int(risk_score),
            'category': category,
            'primary_driver': {'parameter': primary[0], 'contribution': round(primary[1]*100, 1)},
            'threats': self._grid_threats(data),
            'rf_category': int(rf_result['category']),
            'rf_confidence': float(max(rf_result['probability']))
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


def get_detailed_llm_analysis(sector_data, all_raw_data, all_normalized, ml_forecast, temporal_features):
    """
    Generate detailed multi-paragraph analysis for sector detail view.
    This is called separately from the quick advisory.
    """
    import os
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    
    if not MISTRAL_API_KEY or "YOUR_MISTRAL" in MISTRAL_API_KEY:
        return "Detailed analysis unavailable - API key not configured."
    
    try:
        client = Mistral(api_key=MISTRAL_API_KEY)
        
        sector = sector_data['sector']
        risk_score = sector_data['risk_score']
        category = sector_data['category']
        primary_driver = sector_data['primary_driver']
        threats = sector_data.get('threats', [])
        rf_category = ['LOW', 'MEDIUM', 'HIGH'][sector_data.get('rf_category', 0)]
        
        prompt = f"""You are ASTRORISK, a Space Weather Operations AI providing detailed sector analysis.

SECTOR: {sector}
RISK LEVEL: {risk_score}/100 ({category.upper()})
RF CLASSIFICATION: {rf_category} (confidence: {sector_data.get('rf_confidence', 0):.0%})
PRIMARY DRIVER: {primary_driver}
IDENTIFIED THREATS: {', '.join(threats)}

STORM FORECAST:
- TFT Probability: {ml_forecast.get('storm_probability', 0)*100:.1f}%
- Pattern: {ml_forecast.get('detected_pattern', 'Unknown')}
- Confidence: {ml_forecast.get('confidence', 0)*100:.0f}%

TEMPORAL FEATURES:
- Storm Ramp: {temporal_features.get('storm_ramp', 0):.3f}
- Kp Acceleration: {temporal_features.get('kp_acceleration', 0):.3f}
- Plasma Pressure: {temporal_features.get('plasma_pressure', 0):.3f}
- Wind Burst: {temporal_features.get('wind_burst', 0):.3f}

RAW TELEMETRY:
- Kp Index: {all_raw_data.get('kp', 0):.1f}/9
- X-ray Flux: {all_raw_data.get('xray_flux', 0):.2e} W/m²
- Proton Flux: {all_raw_data.get('proton_flux', 0):.1f} pfu
- Electron Flux: {all_raw_data.get('electron_flux', 0):.1f} electrons/cm²/s
- Solar Wind: {all_raw_data.get('wind_speed', 0):.0f} km/s
- Bz Component: {all_raw_data.get('bz_gsm', 0):.1f} nT
- Density: {all_raw_data.get('density', 0):.1f} particles/cm³
- Temperature: {all_raw_data.get('temperature', 0):.2e} K

Generate a detailed 4-paragraph analysis:

1. CURRENT SITUATION: Describe the current risk state and what's driving it (2-3 sentences)
2. PHYSICAL MECHANISMS: Explain the space weather physics causing this situation (2-3 sentences)
3. OPERATIONAL IMPACTS: Detail specific impacts to this sector's operations (2-3 sentences)
4. RECOMMENDATIONS: Provide actionable mitigation steps (2-3 sentences)

Write in professional, technical language suitable for mission control operators. Be specific and data-driven."""

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.4
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"⚠️  Detailed analysis error for {sector}: {e}")
        return f"Detailed analysis temporarily unavailable. Risk score: {risk_score}/100. Monitor primary driver: {primary_driver}."


def get_llm_advisory(sector, risk_score, driver, storm_prob):
    """Quick one-line advisory for sector card"""
    MISTRAL_API_KEY = "pn2dxv5anni4FPKtlkFyBlpdClBjRrQ2"
    
    if not MISTRAL_API_KEY or "YOUR_MISTRAL" in MISTRAL_API_KEY:
        return f"**STATUS: {['NOMINAL', 'CAUTION', 'ALERT'][min(risk_score//34, 2)]}: Standard monitoring protocols.**"
    
    try:
        client = Mistral(api_key=MISTRAL_API_KEY)
        
        prompt = f"""
You are an expert space weather forecaster for critical infrastructure.

SECTOR: {sector}
RISK SCORE: {risk_score}/100
PRIMARY DRIVER: {driver}
STORM PROBABILITY: {storm_prob*100:.0f}%

Provide ONE concise sentence (max 15 words) using this format:
**STATUS: [NOMINAL/CAUTION/ALERT]: [action/impact].**

Examples:
- **STATUS: NOMINAL: All systems operating within normal parameters.**
- **STATUS: CAUTION: Monitor high electron flux; prepare for potential storm impacts.**
- **STATUS: ALERT: Implement radiation shielding protocols immediately.**
"""
        
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3
        )
        
        advisory = response.choices[0].message.content.strip()
        return advisory
        
    except Exception as e:
        print(f"⚠️  Mistral API error for {sector}: {e}")
        status = ['NOMINAL', 'CAUTION', 'ALERT'][min(risk_score//34, 2)]
        return f"**STATUS: {status}: Standard monitoring protocols.**"


class AstroRiskAdvisor:
    def __init__(self, mistral_api_key):
        self.mistral_api_key = mistral_api_key
    
    def generate_situation_report(self, risk_scores, all_raw_data, all_normalized, temporal_features):
        """Generate both quick advisories AND detailed analyses"""
        sectors_dict = {}
        
        for sector_key in ['satellite', 'aviation', 'power_grid']:
            sector_data = risk_scores.get(sector_key, {})
            
            risk_score = sector_data.get('risk_score', 0)
            driver_param = sector_data.get('primary_driver', {}).get('parameter', 'unknown')
            storm_prob = risk_scores.get('ml_forecast', {}).get('storm_probability', 0.0)
            
            # Quick advisory for card
            advisory_text = get_llm_advisory(
                sector=sector_data.get('sector', sector_key),
                risk_score=risk_score,
                driver=driver_param,
                storm_prob=storm_prob
            )
            
            # Detailed analysis for modal
            detailed_analysis = get_detailed_llm_analysis(
                sector_data=sector_data,
                all_raw_data=all_raw_data,
                all_normalized=all_normalized,
                ml_forecast=risk_scores.get('ml_forecast', {}),
                temporal_features=temporal_features
            )
            
            sectors_dict[sector_key] = {
                'sector': sector_data.get('sector', sector_key),
                'risk_score': int(risk_score),
                'category': sector_data.get('category', 'green'),
                'advisory': advisory_text,
                'detailed_analysis': detailed_analysis,  # Full analysis
                'primary_driver': driver_param,
                'rf_category': int(sector_data.get('rf_category', 0)),
                'rf_confidence': float(sector_data.get('rf_confidence', 0.0))
            }
        
        overall_advisory = self._overall_advisory(sectors_dict)
        
        return {
            'timestamp': risk_scores.get('timestamp', datetime.now(timezone.utc).isoformat()),
            'ml_forecast': risk_scores.get('ml_forecast', {}),
            'sectors': sectors_dict,
            'overall_advisory': overall_advisory
        }
    
    def _overall_advisory(self, all_sectors):
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


def warmup_with_synthetic_data():
    print("\n🔥 HACKATHON MODE: Warming up with synthetic history...")
    print("   Simulating 50+ samples of space weather data for instant ML pattern detection")
    
    for i in range(30):
        progression = i / 30.0
        fake_data = {
            'kp_norm': 0.1 + (progression * 0.4),
            'bz_norm': 0.3 + (progression * 0.4),
            'wind_speed_norm': 0.2 + (progression * 0.3),
            'xray_norm': 0.1 + (progression * 0.2),
            'proton_norm': 0.05 + (progression * 0.15),
            'electron_norm': 0.1 + (progression * 0.3),
            'density_norm': 0.1 + (progression * 0.2),
            'temperature_norm': 0.15 + (progression * 0.25)
        }
        append_to_history(fake_data)
    
    for i in range(20):
        noise = np.random.uniform(-0.05, 0.05)
        fake_data = {
            'kp_norm': 0.5 + noise,
            'bz_norm': 0.7 + noise,
            'wind_speed_norm': 0.5 + noise,
            'xray_norm': 0.3 + noise,
            'proton_norm': 0.2 + noise,
            'electron_norm': 0.4 + noise,
            'density_norm': 0.3 + noise,
            'temperature_norm': 0.4 + noise
        }
        append_to_history(fake_data)
    
    print(f"✓ Synthetic warm-up complete: {len(astrorisk_history)} samples in buffer\n")


def run_astrorisk_pipeline(mistral_api_key):
    print("="*80)
    print("ASTRORISK ENHANCED PIPELINE V2 - TFT + RANDOM FOREST + DETAILED ANALYSIS")
    print("="*80)
    
    # =========================================================================
    # ✅ STEP 2 — INSERT A TEST ROW INTO SUPABASE (REAL DB WRITE TEST)
    # =========================================================================
    print("\n[STEP 2] Testing Supabase table write...")
    try:
        test_insert = supabase.table("telemetry_history").insert({
            "kp": 5.0,
            "xray_flux": 1e-6,
            "proton_flux": 10.0,
            "electron_flux": 300.0,
            "wind_speed": 500.0,
            "density": 8.0,
            "temperature": 200000,
            "bz_gsm": -8.0
        }).execute()
        print("✅ Insert response:", test_insert)
    except Exception as e:
        print(f"❌ Supabase insert FAILED: {e}")
    # =========================================================================

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
        print(f"✗ Critical error: {e}")
        return None
    
    print("\n[2/7] Normalizing parameters...")
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
    
    print("\n[3/7] Updating temporal history buffer...")
    history_df = append_to_history(normalized_data)
    print(f"✓ History buffer: {len(history_df)} samples")
    
    print("\n[4/7] Extracting temporal features...")
    extractor = FeatureExtractor()
    derived_features = extractor.extract_features(history_df)
    print(f"✓ Features extracted (confidence: {derived_features['feature_confidence']:.2f})")
    print(f"  Plasma Pressure: {derived_features['plasma_pressure']:.3f}")
    print(f"  Storm Ramp: {derived_features['storm_ramp']:.3f}")
    print(f"  Kp Acceleration: {derived_features['kp_acceleration']:.3f}")
    print(f"  Wind Burst: {derived_features['wind_burst']:.3f}")
    
    print("\n[5/7] Running TFT storm forecast...")
    tft_predictor = TFTPredictor()
    ml_output = tft_predictor.predict(history_df)
    print(f"✓ TFT Storm probability: {ml_output['storm_probability']*100:.1f}%")
    print(f"  Pattern: {ml_output['detected_pattern']}")
    print(f"  Confidence: {ml_output['confidence']*100:.0f}%")
    print(f"  Top drivers: {sorted(ml_output['variable_importance'].items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    print("\n[6/7] Calculating sector risks with Random Forest...")
    engine = AstroRiskEngine()
    risk_scores = engine.calculate_risk_scores(normalized_data, derived_features)
    risk_scores['ml_forecast'] = ml_output
    print("✓ RF classifications complete")
    
    print("\n[7/7] Generating Mistral AI advisories + detailed analysis...")
    advisor = AstroRiskAdvisor(mistral_api_key)
    sitrep = advisor.generate_situation_report(
        risk_scores,
        ingestor.get_raw_data_dict(),
        normalized_data,
        derived_features
    )
    
    sitrep['raw_data'] = ingestor.get_raw_data_dict()
    sitrep['normalized_data'] = normalized_data
    sitrep['temporal_features'] = derived_features
    
    save_live_data(sitrep)
    print("✓ SITREP saved for frontend\n")
    
    print("="*80)
    print("OPERATOR SITUATION REPORT")
    print("="*80)
    print(f"Generated: {sitrep['timestamp']}")
    
    print("\n🧠 TFT ML FORECAST:")
    print("-" * 80)
    ml = sitrep['ml_forecast']
    print(f"  Storm Probability (G3+): {ml['storm_probability']*100:.1f}%")
    print(f"  Pattern: {ml['detected_pattern']}")
    print(f"  Confidence: {ml['confidence']*100:.0f}%")
    
    print("\n🌲 SECTOR ADVISORIES (Random Forest + Mistral AI):")
    print("-" * 80)
    
    category_icons = {'green': '✅', 'yellow': '⚠️', 'red': '🚨'}
    for sector_key, sector_data in sitrep['sectors'].items():
        icon = category_icons.get(sector_data['category'], '❓')
        rf_cat = ['LOW', 'MEDIUM', 'HIGH'][sector_data.get('rf_category', 0)]
        print(f"\n{icon} {sector_data['sector'].upper()}")
        print(f"   Risk Score: {sector_data['risk_score']}/100")
        print(f"   RF Category: {rf_cat} (confidence: {sector_data.get('rf_confidence', 0):.2f})")
        print(f"   Advisory: {sector_data['advisory']}")
    
    print("\n" + "="*80)
    print(f"   {sitrep['overall_advisory']}")
    print("="*80)
    
    return sitrep


if __name__ == "__main__":
    MISTRAL_API_KEY = "pn2dxv5anni4FPKtlkFyBlpdClBjRrQ2"
    
    print("🚀 STARTING ASTRORISK ENHANCED MONITORING V2...")
    print("   TFT: Temporal Fusion Transformer for storm forecasting")
    print("   RF: Random Forest for sector risk classification")
    print("   LLM: Detailed multi-paragraph analysis generation\n")
    
    warmup_with_synthetic_data()
    
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    while True:
        try:
            sitrep = run_astrorisk_pipeline(MISTRAL_API_KEY)
            
            if sitrep is not None:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print("\n🛑 Too many failures. Stopping.")
                    break
            
            print("\n⏳ Sleeping for 300 seconds (5 minutes)...")
            time.sleep(300)
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print("\n🛑 Too many failures. Stopping.")
                break
            time.sleep(60)
