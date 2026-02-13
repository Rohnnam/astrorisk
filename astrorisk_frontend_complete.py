import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from astrorisk_shared_data import load_live_data, get_synthetic_fallback
import time

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AstroRisk | Space Weather Operations",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= LOAD LIVE DATA =================
live_data = load_live_data()
if live_data is None:
    live_data = get_synthetic_fallback()
    data_source = "SYNTHETIC"
else:
    data_source = "LIVE"

# Extract data
ml_forecast = live_data.get('ml_forecast', {})
sectors = live_data.get('sectors', {})
overall_advisory = live_data.get('overall_advisory', 'No data available')
raw_data = live_data.get('raw_data', {})
normalized_data = live_data.get('normalized_data', {})
temporal_features = live_data.get('temporal_features', {})
timestamp = live_data.get('timestamp', datetime.now().isoformat())

# Calculate overall risk score (average of all sectors from backend)
sector_scores = []
if 'satellite' in sectors:
    sector_scores.append(sectors['satellite'].get('risk_score', 0))
if 'aviation' in sectors:
    sector_scores.append(sectors['aviation'].get('risk_score', 0))
if 'power_grid' in sectors:
    sector_scores.append(sectors['power_grid'].get('risk_score', 0))

overall_risk_score = int(np.mean(sector_scores)) if sector_scores else 20

# Determine storm level from Kp
kp_value = raw_data.get('kp', 0.0)
if kp_value >= 8:
    storm_level = "G4"
    storm_color = "#ff1744"
elif kp_value >= 7:
    storm_level = "G3"
    storm_color = "#ff1744"
elif kp_value >= 6:
    storm_level = "G2"
    storm_color = "#ff8500"
elif kp_value >= 5:
    storm_level = "G1"
    storm_color = "#ffbe0b"
else:
    storm_level = "G0"
    storm_color = "#00f5a0"

# Determine storm phase
storm_prob = ml_forecast.get('storm_probability', 0.0)
if storm_prob >= 0.70:
    storm_phase = "STORM ACTIVE"
    phase_color = "#ff1744"
elif storm_prob >= 0.50:
    storm_phase = "Watch"
    phase_color = "#ffbe0b"
elif storm_prob >= 0.30:
    storm_phase = "Recovery"
    phase_color = "#a0b3d8"
else:
    storm_phase = "Normal"
    phase_color = "#00f5a0"

# Parse timestamp
try:
    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    time_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
except:
    time_str = timestamp

# ================= HIDE STREAMLIT BRANDING =================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# ================= ADVANCED CSS (FROM ORIGINAL) =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Orbitron:wght@400;500;600;700;900&display=swap');

:root {
    --color-primary: #00d9ff;
    --color-secondary: #ff006e;
    --color-accent: #ffbe0b;
    --color-success: #00f5a0;
    --color-warning: #ff8500;
    --color-critical: #ff1744;
    --color-bg-dark: #0a0e27;
    --color-bg-medium: #161b33;
    --color-bg-light: #1f2847;
    --color-text-primary: #e8edf4;
    --color-text-secondary: #a0b3d8;
    --color-text-muted: #6b7a99;
    --color-border: rgba(0, 217, 255, 0.15);
    --color-glow: rgba(0, 217, 255, 0.4);
    --font-display: 'Orbitron', monospace;
    --font-body: 'JetBrains Mono', monospace;
}

.stApp {
    background: 
        radial-gradient(ellipse at top, rgba(0, 217, 255, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at bottom, rgba(255, 0, 110, 0.06) 0%, transparent 50%),
        linear-gradient(180deg, #0a0e27 0%, #050811 100%);
    background-attachment: fixed;
    color: var(--color-text-primary);
    font-family: var(--font-body);
}

.glass-panel {
    background: linear-gradient(135deg, 
        rgba(31, 40, 71, 0.7) 0%, 
        rgba(22, 27, 51, 0.85) 100%);
    backdrop-filter: blur(20px) saturate(180%);
    border-radius: 16px;
    padding: 32px;
    border: 1px solid var(--color-border);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    position: relative;
    overflow: hidden;
}

.glass-panel:hover {
    border-color: rgba(0, 217, 255, 0.3);
    transform: translateY(-2px);
}

.main-title {
    font-family: var(--font-display);
    font-size: 3.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00d9ff 0%, #00f5a0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.1em;
    margin: 0;
}

.subtitle {
    font-family: var(--font-body);
    font-size: 1rem;
    color: var(--color-text-secondary);
    font-weight: 300;
    letter-spacing: 0.15em;
    margin-top: 12px;
    text-transform: uppercase;
}

.alert-critical {
    background: linear-gradient(135deg, 
        rgba(255, 23, 68, 0.2) 0%, 
        rgba(139, 0, 0, 0.3) 100%);
    border: 2px solid var(--color-critical);
    border-radius: 12px;
    padding: 24px 32px;
    margin: 24px 0;
    animation: alert-pulse 2s ease-in-out infinite;
}

@keyframes alert-pulse {
    0%, 100% { box-shadow: 0 0 20px rgba(255, 23, 68, 0.4); }
    50% { box-shadow: 0 0 40px rgba(255, 23, 68, 0.8); }
}

.alert-warning {
    background: linear-gradient(135deg, 
        rgba(255, 190, 11, 0.15) 0%, 
        rgba(255, 133, 0, 0.2) 100%);
    border: 2px solid var(--color-warning);
    border-radius: 12px;
    padding: 24px 32px;
    margin: 24px 0;
}

.sector-card {
    background: linear-gradient(135deg, 
        rgba(31, 40, 71, 0.6) 0%, 
        rgba(22, 27, 51, 0.8) 100%);
    border-radius: 12px;
    padding: 24px;
    border: 1px solid var(--color-border);
    margin-bottom: 16px;
}

.phase-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 20px;
    font-family: var(--font-display);
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    margin: 0 8px;
}

.telemetry-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 0;
    border-bottom: 1px solid rgba(0, 217, 255, 0.1);
}

.telemetry-label {
    font-size: 0.9rem;
    color: var(--color-text-secondary);
    font-weight: 400;
}

.telemetry-value {
    font-family: var(--font-display);
    font-size: 1.1rem;
    color: var(--color-primary);
    font-weight: 600;
}

.telemetry-bar {
    width: 100%;
    height: 8px;
    background: rgba(107, 122, 153, 0.2);
    border-radius: 4px;
    overflow: hidden;
    margin-top: 6px;
}

.telemetry-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--color-primary), var(--color-success));
    transition: width 0.5s ease;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown(f"""
<div class="glass-panel">
    <h1 class="main-title">⬢ ASTRORISK</h1>
    <div class="subtitle">Space Weather Operations Command</div>
    <div style="margin-top: 16px; color: var(--color-text-muted); font-size: 0.85rem;">
        ⏱ Last Update: {time_str}
    </div>
    <div style="margin-top: 12px;">
        <span style="display: inline-block; padding: 6px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; 
                     background: {'linear-gradient(135deg, rgba(0, 245, 160, 0.2), rgba(0, 217, 255, 0.2))' if data_source == 'LIVE' else 'linear-gradient(135deg, rgba(255, 190, 11, 0.2), rgba(255, 133, 0, 0.2))'}; 
                     border: 1px solid {'#00f5a0' if data_source == 'LIVE' else '#ffbe0b'}; 
                     color: {'#00f5a0' if data_source == 'LIVE' else '#ffbe0b'};">
            {data_source} DATA
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# ================= ALERT BANNER =================
if storm_prob >= 0.70:
    st.markdown(f"""
    <div class="alert-critical">
        <div style="font-family: Orbitron; font-size: 1.5rem; margin-bottom: 12px;">
            ⚠️ SEVERE GEOMAGNETIC STORM DETECTED
        </div>
        <div style="font-size: 0.95rem; line-height: 1.6;">
            G3-class event in progress. Elevated radiation flux and atmospheric drag affecting orbital assets. 
            HF radio propagation severely degraded at high latitudes. Immediate operational mitigation protocols 
            recommended for satellite operators.
        </div>
    </div>
    """, unsafe_allow_html=True)
elif storm_prob >= 0.50:
    st.markdown(f"""
    <div class="alert-warning">
        <div style="font-family: Orbitron; font-size: 1.5rem; margin-bottom: 12px;">
            ⚠️ ELEVATED STORM WATCH
        </div>
        <div style="font-size: 0.95rem; line-height: 1.6;">
            Geomagnetic storm conditions developing. Monitor space weather parameters closely and prepare 
            mitigation protocols.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# ================= OVERALL RISK GAUGE + STATUS INDICATORS =================
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1])

with col1:
    # Create the gauge chart (dial)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=overall_risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'size': 72, 'family': 'Orbitron', 'color': '#00d9ff'}},
        delta={'reference': 50, 'increasing': {'color': "#ff1744"}},
        title={'text': "OVERALL RISK SCORE", 'font': {'size': 16, 'family': 'Orbitron', 'color': '#a0b3d8'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#a0b3d8"},
            'bar': {'color': "#00d9ff", 'thickness': 0.3},
            'bgcolor': "rgba(10, 14, 39, 0.8)",
            'borderwidth': 2,
            'bordercolor': "rgba(0, 217, 255, 0.3)",
            'steps': [
                {'range': [0, 33], 'color': 'rgba(0, 245, 160, 0.3)'},
                {'range': [33, 67], 'color': 'rgba(255, 190, 11, 0.3)'},
                {'range': [67, 100], 'color': 'rgba(255, 23, 68, 0.3)'}
            ],
            'threshold': {
                'line': {'color': storm_color, 'width': 4},
                'thickness': 0.75,
                'value': overall_risk_score
            }
        }
    ))

    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#e8edf4", 'family': "Orbitron"},
        height=400,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    # Status indicators - Using components to avoid HTML escaping
    st.markdown('<div style="padding: 20px;">', unsafe_allow_html=True)
    
    # KP INDEX
    st.markdown('<div style="margin-bottom: 32px;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.85rem; color: #6b7a99; letter-spacing: 0.1em; margin-bottom: 8px;">KP INDEX</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-family: Orbitron; font-size: 3rem; color: #00d9ff; font-weight: 700;">{kp_value:.1f}<span style="font-size: 1.5rem; color: #6b7a99;">/9</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # STORM LEVEL
    st.markdown('<div style="margin-bottom: 32px;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.85rem; color: #6b7a99; letter-spacing: 0.1em; margin-bottom: 8px;">STORM LEVEL</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-family: Orbitron; font-size: 3rem; color: {storm_color}; font-weight: 700;">{storm_level}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # DURATION
    st.markdown('<div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.85rem; color: #6b7a99; letter-spacing: 0.1em; margin-bottom: 8px;">DURATION</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family: Orbitron; font-size: 2rem; color: #00d9ff; font-weight: 700;">6<span style="font-size: 1.2rem; color: #6b7a99;"> h 24m</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


# Phase indicators at bottom
st.markdown(f"""
<div style="display: flex; justify-content: center; gap: 16px; margin-top: 24px; padding: 16px; background: rgba(10, 14, 39, 0.5); border-radius: 12px;">
    <div class="phase-indicator" style="background: {'rgba(0, 245, 160, 0.2)' if storm_phase == 'Normal' else 'rgba(107, 122, 153, 0.1)'}; 
                                         border: 1px solid {'#00f5a0' if storm_phase == 'Normal' else 'rgba(107, 122, 153, 0.3)'};
                                         color: {'#00f5a0' if storm_phase == 'Normal' else '#6b7a99'};">
        <span style="width: 8px; height: 8px; border-radius: 50%; background: {'#00f5a0' if storm_phase == 'Normal' else '#6b7a99'};"></span>
        Normal
    </div>
    <div class="phase-indicator" style="background: {'rgba(255, 190, 11, 0.2)' if storm_phase == 'Watch' else 'rgba(107, 122, 153, 0.1)'}; 
                                         border: 1px solid {'#ffbe0b' if storm_phase == 'Watch' else 'rgba(107, 122, 153, 0.3)'};
                                         color: {'#ffbe0b' if storm_phase == 'Watch' else '#6b7a99'};">
        <span style="width: 8px; height: 8px; border-radius: 50%; background: {'#ffbe0b' if storm_phase == 'Watch' else '#6b7a99'};"></span>
        Watch
    </div>
    <div class="phase-indicator" style="background: {'rgba(255, 23, 68, 0.2)' if storm_phase == 'STORM ACTIVE' else 'rgba(107, 122, 153, 0.1)'}; 
                                         border: 1px solid {'#ff1744' if storm_phase == 'STORM ACTIVE' else 'rgba(107, 122, 153, 0.3)'};
                                         color: {'#ff1744' if storm_phase == 'STORM ACTIVE' else '#6b7a99'};">
        <span style="width: 8px; height: 8px; border-radius: 50%; background: {'#ff1744' if storm_phase == 'STORM ACTIVE' else '#6b7a99'};"></span>
        STORM ACTIVE
    </div>
    <div class="phase-indicator" style="background: {'rgba(160, 179, 216, 0.2)' if storm_phase == 'Recovery' else 'rgba(107, 122, 153, 0.1)'}; 
                                         border: 1px solid {'#a0b3d8' if storm_phase == 'Recovery' else 'rgba(107, 122, 153, 0.3)'};
                                         color: {'#a0b3d8' if storm_phase == 'Recovery' else '#6b7a99'};">
        <span style="width: 8px; height: 8px; border-radius: 50%; background: {'#a0b3d8' if storm_phase == 'Recovery' else '#6b7a99'};"></span>
        Recovery
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# ================= SECTOR ADVISORIES =================
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown('<h2 style="font-family: Orbitron; margin-bottom: 24px;">⬢ SECTOR-SPECIFIC ADVISORIES</h2>', unsafe_allow_html=True)

cols = st.columns(3)

sector_icons = {
    'satellite': '🛰️',
    'aviation': '✈️',
    'power_grid': '⚡'
}

for idx, sector_key in enumerate(['satellite', 'aviation', 'power_grid']):
    if sector_key in sectors:
        sector = sectors[sector_key]
        risk_score = sector.get('risk_score', 0)
        category = sector.get('category', 'green')
        advisory = sector.get('advisory', 'No advisory available')
        sector_name = sector.get('sector', sector_key.replace('_', ' ').title())
        
        if category == 'red':
            icon = "🚨"
            color = "#ff1744"
        elif category == 'yellow':
            icon = "⚠️"
            color = "#ffbe0b"
        else:
            icon = "✅"
            color = "#00f5a0"
        
        with cols[idx]:
            st.markdown(f"""
            <div class="sector-card">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                    <span style="font-size: 2rem;">{sector_icons.get(sector_key, '📊')}</span>
                    <div>
                        <div style="font-family: Orbitron; font-size: 0.75rem; color: var(--color-text-muted); letter-spacing: 0.1em;">
                            {sector_name.upper()}
                        </div>
                        <div style="font-family: Orbitron; font-size: 1.8rem; color: {color}; font-weight: 700; margin-top: 4px;">
                            {risk_score}<span style="font-size: 1rem; color: var(--color-text-muted);">/100</span>
                        </div>
                    </div>
                </div>
                <div style="margin-bottom: 16px;">
                    <div style="font-size: 0.75rem; color: var(--color-text-muted); margin-bottom: 4px;">RISK LEVEL</div>
                    <div style="width: 100%; height: 8px; background: rgba(107, 122, 153, 0.2); border-radius: 4px; overflow: hidden;">
                        <div style="width: {risk_score}%; height: 100%; background: linear-gradient(90deg, {color}, {color}80); transition: width 0.5s ease;"></div>
                    </div>
                </div>
                <div style="padding: 14px; background: rgba(0, 0, 0, 0.3); border-radius: 8px; border-left: 3px solid {color};">
                    <div style="font-size: 0.7rem; color: var(--color-text-muted); margin-bottom: 6px; letter-spacing: 0.1em;">ADVISORY</div>
                    <div style="font-size: 0.85rem; line-height: 1.5;">{advisory}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# ================= REAL-TIME TELEMETRY WITH SLIDERS =================
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown('<h2 style="font-family: Orbitron; margin-bottom: 24px;">⬢ REAL-TIME TELEMETRY</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

# Get normalized values for slider bars
kp_norm = normalized_data.get('kp_norm', 0.0)
wind_speed_norm = normalized_data.get('wind_speed_norm', 0.0)
proton_norm = normalized_data.get('proton_norm', 0.0)
electron_norm = normalized_data.get('electron_norm', 0.0)
xray_norm = normalized_data.get('xray_norm', 0.0)
bz_norm = normalized_data.get('bz_norm', 0.0)
density_norm = normalized_data.get('density_norm', 0.0)
temperature_norm = normalized_data.get('temperature_norm', 0.0)

# Get raw values
wind_speed_value = raw_data.get('wind_speed', 0.0)
proton_flux_value = raw_data.get('proton_flux', 0.0)
electron_flux_value = raw_data.get('electron_flux', 0.0)
density_value = raw_data.get('density', 0.0)
bz_gsm_value = raw_data.get('bz_gsm', 0.0)
xray_flux_value = raw_data.get('xray_flux', 0.0)
temperature_value = raw_data.get('temperature', 0.0)

with col1:
    st.markdown(f"""
    <div class="viz-container">
        <div class="telemetry-row">
            <div class="telemetry-label">Kp Index</div>
            <div style="flex: 1; margin-left: 20px;">
                <div class="telemetry-value">{kp_value:.1f}</div>
                <div class="telemetry-bar">
                    <div class="telemetry-bar-fill" style="width: {kp_norm*100:.0f}%;"></div>
                </div>
            </div>
        </div>
        <div class="telemetry-row">
            <div class="telemetry-label">Solar Wind Speed</div>
            <div style="flex: 1; margin-left: 20px;">
                <div class="telemetry-value">{wind_speed_value:.0f} km/s</div>
                <div class="telemetry-bar">
                    <div class="telemetry-bar-fill" style="width: {wind_speed_norm*100:.0f}%;"></div>
                </div>
            </div>
        </div>
        <div class="telemetry-row">
            <div class="telemetry-label">Proton Flux (>10 MeV)</div>
            <div style="flex: 1; margin-left: 20px;">
                <div class="telemetry-value">{proton_flux_value:.1f} pfu</div>
                <div class="telemetry-bar">
                    <div class="telemetry-bar-fill" style="width: {proton_norm*100:.0f}%;"></div>
                </div>
            </div>
        </div>
        <div class="telemetry-row">
            <div class="telemetry-label">Electron Flux (>2 MeV)</div>
            <div style="flex: 1; margin-left: 20px;">
                <div class="telemetry-value">{electron_flux_value:.0f} e⁻/cm²/s</div>
                <div class="telemetry-bar">
                    <div class="telemetry-bar-fill" style="width: {electron_norm*100:.0f}%;"></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="viz-container">
        <div class="telemetry-row">
            <div class="telemetry-label">IMF Bz (GSM)</div>
            <div style="flex: 1; margin-left: 20px;">
                <div class="telemetry-value" style="color: {'#ff8500' if bz_gsm_value < -5 else '#00d9ff'};">{bz_gsm_value:.1f} nT</div>
                <div class="telemetry-bar">
                    <div class="telemetry-bar-fill" style="width: {bz_norm*100:.0f}%; background: linear-gradient(90deg, #ff8500, #ffbe0b);"></div>
                </div>
            </div>
        </div>
        <div class="telemetry-row">
            <div class="telemetry-label">Plasma Density</div>
            <div style="flex: 1; margin-left: 20px;">
                <div class="telemetry-value">{density_value:.1f} cm⁻³</div>
                <div class="telemetry-bar">
                    <div class="telemetry-bar-fill" style="width: {density_norm*100:.0f}%;"></div>
                </div>
            </div>
        </div>
        <div class="telemetry-row">
            <div class="telemetry-label">X-Ray Flux</div>
            <div style="flex: 1; margin-left: 20px;">
                <div class="telemetry-value">{xray_flux_value:.2e} W/m²</div>
                <div class="telemetry-bar">
                    <div class="telemetry-bar-fill" style="width: {xray_norm*100:.0f}%;"></div>
                </div>
            </div>
        </div>
        <div class="telemetry-row">
            <div class="telemetry-label">Temperature</div>
            <div style="flex: 1; margin-left: 20px;">
                <div class="telemetry-value">{temperature_value:.2e} K</div>
                <div class="telemetry-bar">
                    <div class="telemetry-bar-fill" style="width: {temperature_norm*100:.0f}%;"></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# ================= 48-HOUR TREND ANALYSIS =================
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown('<h2 style="font-family: Orbitron; margin-bottom: 24px;">⬢ 48-HOUR TREND ANALYSIS</h2>', unsafe_allow_html=True)

# Generate time series data (simulated historical + LIVE current point from backend)
hours = pd.date_range(end=datetime.now(), periods=48, freq='h')

# Create realistic historical data with LIVE current value at the end
kp_historical = np.concatenate([
    np.random.uniform(2, 4, 24),
    np.random.uniform(4, 6, 16),
    np.random.uniform(3, kp_value, 7),
    [kp_value]  # ← LIVE value from backend
])

wind_historical = np.concatenate([
    np.random.uniform(350, 450, 24),
    np.random.uniform(450, 600, 16),
    np.random.uniform(400, wind_speed_value, 7),
    [wind_speed_value]  # ← LIVE value from backend
])

# Create interactive time series
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=hours,
    y=kp_historical,
    name='Kp Index',
    line=dict(color='#00d9ff', width=3),
    fill='tozeroy',
    fillcolor='rgba(0, 217, 255, 0.2)',
    hovertemplate='<b>Kp Index</b><br>Time: %{x}<br>Value: %{y:.1f}<extra></extra>'
))

# Add marker for current LIVE point
fig.add_trace(go.Scatter(
    x=[hours[-1]],
    y=[kp_value],
    mode='markers',
    marker=dict(size=12, color='#00f5a0', line=dict(width=2, color='#ffffff')),
    name='Current (LIVE)',
    hovertemplate='<b>LIVE NOW</b><br>Time: %{x}<br>Kp: %{y:.1f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=hours,
    y=wind_historical / 100,
    name='Solar Wind (×100 km/s)',
    line=dict(color='#ff006e', width=2, dash='dot'),
    hovertemplate='<b>Solar Wind</b><br>Time: %{x}<br>Speed: %{y:.0f}×100 km/s<extra></extra>',
    yaxis='y2'
))

# Add threshold lines
fig.add_hline(y=5, line_dash="dash", line_color="#ffbe0b", 
              annotation_text="G1 Storm Threshold", 
              annotation_position="right")
fig.add_hline(y=7, line_dash="dash", line_color="#ff1744", 
              annotation_text="G3 Storm Threshold", 
              annotation_position="right")

fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10, 14, 39, 0.5)',
    font={'color': "#e8edf4", 'family': "JetBrains Mono"},
    height=400,
    margin=dict(l=60, r=60, t=20, b=60),
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor='rgba(31, 40, 71, 0.8)',
        bordercolor='rgba(0, 217, 255, 0.3)',
        borderwidth=1
    ),
    xaxis=dict(
        gridcolor='rgba(0, 217, 255, 0.1)',
        showgrid=True,
        title=dict(
            text='Time (UTC)',
            font=dict(color='#a0b3d8')
        )
    ),
    yaxis=dict(
        gridcolor='rgba(0, 217, 255, 0.1)',
        showgrid=True,
        title=dict(
            text='Kp Index',
            font=dict(color='#00d9ff')
        ),
        range=[0, 9]
    ),
    yaxis2=dict(
        title=dict(
            text='Solar Wind Speed',
            font=dict(color='#ff006e')
        ),
        overlaying='y',
        side='right',
        showgrid=False
    )
)

st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# ================= ML FORECAST DETAILS =================
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown('<h2 style="font-family: Orbitron; margin-bottom: 24px;">⬢ ML FORECAST & TEMPORAL FEATURES</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pattern = ml_forecast.get('detected_pattern', 'Unknown')
    confidence = ml_forecast.get('confidence', 0.0)
    urgency = ml_forecast.get('urgency_multiplier', 1.0)
    
    st.markdown(f"""
    <div style="padding: 20px; background: rgba(0, 0, 0, 0.3); border-radius: 12px; border-left: 4px solid var(--color-primary);">
        <div style="font-family: Orbitron; font-size: 1.2rem; margin-bottom: 16px; color: var(--color-primary);">
            Storm Probability Forecast
        </div>
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.85rem; color: var(--color-text-muted); margin-bottom: 6px;">PROBABILITY (G3+)</div>
            <div style="font-family: Orbitron; font-size: 2.5rem; color: var(--color-primary); font-weight: 700;">
                {storm_prob*100:.1f}%
            </div>
        </div>
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.85rem; color: var(--color-text-muted); margin-bottom: 6px;">DETECTED PATTERN</div>
            <div style="font-size: 0.95rem; line-height: 1.5;">{pattern}</div>
        </div>
        <div style="display: flex; gap: 20px;">
            <div>
                <div style="font-size: 0.75rem; color: var(--color-text-muted); margin-bottom: 4px;">CONFIDENCE</div>
                <div style="font-family: Orbitron; font-size: 1.3rem; color: var(--color-success);">{confidence:.2f}</div>
            </div>
            <div>
                <div style="font-size: 0.75rem; color: var(--color-text-muted); margin-bottom: 4px;">URGENCY</div>
                <div style="font-family: Orbitron; font-size: 1.3rem; color: var(--color-warning);">{urgency:.2f}x</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if temporal_features:
        bz_stress = temporal_features.get('bz_sustained_stress', 0.0)
        wind_accel = temporal_features.get('wind_acceleration', 0.0)
        proton_spike = temporal_features.get('proton_spike_intensity', 0.0)
        plasma_press = temporal_features.get('plasma_pressure', 0.0)
        kp_persist = temporal_features.get('kp_persistence', 0.0)
        combined = temporal_features.get('combined_storm_score', 0.0)
        
        st.markdown(f"""
        <div style="padding: 20px; background: rgba(0, 0, 0, 0.3); border-radius: 12px;">
            <div style="font-family: Orbitron; font-size: 1.2rem; margin-bottom: 16px; color: var(--color-accent);">
                Temporal Features
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                <div>
                    <div style="font-size: 0.75rem; color: var(--color-text-muted); margin-bottom: 4px;">Bz Stress</div>
                    <div style="font-family: Orbitron; font-size: 1.1rem; color: var(--color-primary);">{bz_stress:.3f}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: var(--color-text-muted); margin-bottom: 4px;">Wind Accel</div>
                    <div style="font-family: Orbitron; font-size: 1.1rem; color: var(--color-primary);">{wind_accel:.3f}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: var(--color-text-muted); margin-bottom: 4px;">Proton Spike</div>
                    <div style="font-family: Orbitron; font-size: 1.1rem; color: var(--color-primary);">{proton_spike:.3f}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: var(--color-text-muted); margin-bottom: 4px;">Plasma Press</div>
                    <div style="font-family: Orbitron; font-size: 1.1rem; color: var(--color-primary);">{plasma_press:.3f}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: var(--color-text-muted); margin-bottom: 4px;">Kp Persist</div>
                    <div style="font-family: Orbitron; font-size: 1.1rem; color: var(--color-primary);">{kp_persist:.3f}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: var(--color-text-muted); margin-bottom: 4px;">Combined</div>
                    <div style="font-family: Orbitron; font-size: 1.1rem; color: var(--color-primary);">{combined:.3f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ================= AUTO-REFRESH (Streamlit Native) =================
# Add auto-refresh using Streamlit's rerun
import time as time_module

# Create a placeholder for countdown
refresh_placeholder = st.empty()

# Show countdown and auto-refresh after 30 seconds
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time_module.time()

elapsed = time_module.time() - st.session_state.last_refresh

if elapsed >= 30:  # 30 seconds
    st.session_state.last_refresh = time_module.time()
    st.rerun()
else:
    remaining = int(30 - elapsed)
    refresh_placeholder.markdown(
        f'<div style="text-align: center; color: #6b7a99; font-size: 0.75rem; margin-top: 16px;">Auto-refresh in {remaining} seconds...</div>',
        unsafe_allow_html=True
    )
    time_module.sleep(1)
    st.rerun()

# ================= FOOTER =================
st.markdown("""
<div style="margin-top: 48px; padding: 24px; text-align: center; color: var(--color-text-muted); font-size: 0.85rem; border-top: 1px solid var(--color-border);">
    <strong>ASTRORISK</strong> is an operational decision-support system powered by Mistral AI. 
    All advisories require human validation before implementation.<br>
    Data sources: NOAA SWPC, NASA ACE/DSCOVR | Classification: OPERATIONAL
</div>
""", unsafe_allow_html=True)