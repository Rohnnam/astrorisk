import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AstroRisk | Space Weather Operations",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= HIDE STREAMLIT BRANDING =================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# ================= ADVANCED CSS DESIGN SYSTEM =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Orbitron:wght@400;500;600;700;900&display=swap');

/* ========== ROOT VARIABLES ========== */
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
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.4);
    --shadow-lg: 0 16px 48px rgba(0, 0, 0, 0.6);
    --shadow-glow: 0 0 20px var(--color-glow);
}

/* ========== GLOBAL STYLES ========== */
.stApp {
    background: 
        radial-gradient(ellipse at top, rgba(0, 217, 255, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at bottom, rgba(255, 0, 110, 0.06) 0%, transparent 50%),
        linear-gradient(180deg, #0a0e27 0%, #050811 100%);
    background-attachment: fixed;
    color: var(--color-text-primary);
    font-family: var(--font-body);
}

/* ========== GRID BACKGROUND ========== */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image: 
        linear-gradient(rgba(0, 217, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 217, 255, 0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}

/* ========== TYPOGRAPHY ========== */
h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-display);
    font-weight: 700;
    letter-spacing: 0.05em;
    color: var(--color-text-primary);
}

/* ========== GLASSMORPHISM CONTAINERS ========== */
.glass-panel {
    background: linear-gradient(135deg, 
        rgba(31, 40, 71, 0.7) 0%, 
        rgba(22, 27, 51, 0.85) 100%);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border-radius: 16px;
    padding: 32px;
    border: 1px solid var(--color-border);
    box-shadow: var(--shadow-md);
    position: relative;
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-panel::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, 
        transparent 0%, 
        var(--color-primary) 50%, 
        transparent 100%);
    opacity: 0.5;
}

.glass-panel:hover {
    border-color: rgba(0, 217, 255, 0.3);
    box-shadow: var(--shadow-lg), var(--shadow-glow);
    transform: translateY(-2px);
}

/* ========== HEADER SECTION ========== */
.header-container {
    background: linear-gradient(135deg, 
        rgba(31, 40, 71, 0.9) 0%, 
        rgba(22, 27, 51, 0.95) 100%);
    backdrop-filter: blur(25px);
    border-radius: 20px;
    padding: 40px 48px;
    border: 1px solid var(--color-border);
    box-shadow: var(--shadow-lg);
    position: relative;
    overflow: hidden;
    margin-bottom: 32px;
}

.header-container::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0, 217, 255, 0.1) 0%, transparent 70%);
    animation: pulse-glow 4s ease-in-out infinite;
    pointer-events: none;
}

@keyframes pulse-glow {
    0%, 100% { opacity: 0.3; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.1); }
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
    text-shadow: 0 0 30px rgba(0, 217, 255, 0.3);
    animation: title-shimmer 3s ease-in-out infinite;
}

@keyframes title-shimmer {
    0%, 100% { filter: brightness(1); }
    50% { filter: brightness(1.2); }
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

.timestamp {
    font-family: var(--font-body);
    font-size: 0.85rem;
    color: var(--color-text-muted);
    font-weight: 400;
    margin-top: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.timestamp::before {
    content: '';
    width: 8px;
    height: 8px;
    background: var(--color-success);
    border-radius: 50%;
    box-shadow: 0 0 10px var(--color-success);
    animation: blink 2s ease-in-out infinite;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ========== ALERT BANNER ========== */
.alert-critical {
    background: linear-gradient(135deg, 
        rgba(255, 23, 68, 0.2) 0%, 
        rgba(139, 0, 0, 0.3) 100%);
    border: 2px solid var(--color-critical);
    border-radius: 12px;
    padding: 24px 32px;
    margin: 24px 0;
    position: relative;
    overflow: hidden;
    animation: alert-pulse 2s ease-in-out infinite;
}

@keyframes alert-pulse {
    0%, 100% { 
        box-shadow: 0 0 20px rgba(255, 23, 68, 0.4),
                    inset 0 0 20px rgba(255, 23, 68, 0.1);
    }
    50% { 
        box-shadow: 0 0 40px rgba(255, 23, 68, 0.8),
                    inset 0 0 30px rgba(255, 23, 68, 0.2);
    }
}

.alert-critical::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, 
        transparent 0%, 
        rgba(255, 255, 255, 0.1) 50%, 
        transparent 100%);
    animation: alert-scan 3s linear infinite;
}

@keyframes alert-scan {
    0% { left: -100%; }
    100% { left: 100%; }
}

.alert-title {
    font-family: var(--font-display);
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--color-critical);
    margin: 0 0 12px 0;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.alert-message {
    font-size: 0.95rem;
    color: var(--color-text-primary);
    line-height: 1.6;
    margin: 0;
}

/* ========== STATUS CARDS ========== */
.status-card {
    background: linear-gradient(135deg, 
        rgba(31, 40, 71, 0.6) 0%, 
        rgba(22, 27, 51, 0.8) 100%);
    border-radius: 16px;
    padding: 28px;
    border-left: 4px solid;
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
    height: 100%;
}

.status-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, 
        rgba(255, 255, 255, 0.05) 0%, 
        transparent 100%);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.status-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
}

.status-card:hover::after {
    opacity: 1;
}

.card-critical {
    border-left-color: var(--color-critical);
}

.card-critical:hover {
    box-shadow: 0 20px 40px rgba(255, 23, 68, 0.3);
}

.card-warning {
    border-left-color: var(--color-warning);
}

.card-warning:hover {
    box-shadow: 0 20px 40px rgba(255, 133, 0, 0.3);
}

.card-safe {
    border-left-color: var(--color-success);
}

.card-safe:hover {
    box-shadow: 0 20px 40px rgba(0, 245, 160, 0.3);
}

.card-title {
    font-family: var(--font-display);
    font-size: 1.25rem;
    font-weight: 600;
    margin: 0 0 16px 0;
    color: var(--color-text-primary);
    display: flex;
    align-items: center;
    gap: 12px;
}

.card-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--color-text-muted);
    font-weight: 600;
    margin: 12px 0 4px 0;
}

.card-value {
    font-size: 0.9rem;
    color: var(--color-text-secondary);
    line-height: 1.5;
    margin: 0 0 12px 0;
}

.card-action {
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 16px;
    padding: 8px 16px;
    border-radius: 6px;
    display: inline-block;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ========== METRIC DISPLAYS ========== */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin: 24px 0;
}

.metric-box {
    background: rgba(31, 40, 71, 0.4);
    border: 1px solid var(--color-border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-box:hover {
    background: rgba(31, 40, 71, 0.6);
    border-color: var(--color-primary);
    transform: translateY(-4px);
}

.metric-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--color-text-muted);
    margin-bottom: 8px;
}

.metric-value {
    font-family: var(--font-display);
    font-size: 2rem;
    font-weight: 700;
    color: var(--color-primary);
    text-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
}

.metric-unit {
    font-size: 0.8rem;
    color: var(--color-text-secondary);
    margin-left: 4px;
}

/* ========== PHASE INDICATOR ========== */
.phase-bar {
    background: rgba(31, 40, 71, 0.5);
    border: 1px solid var(--color-border);
    border-radius: 12px;
    padding: 20px 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}

.phase-item {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.9rem;
    font-weight: 500;
    padding: 8px 16px;
    border-radius: 8px;
    transition: all 0.3s ease;
    cursor: pointer;
}

.phase-item:hover {
    background: rgba(0, 217, 255, 0.1);
}

.phase-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    box-shadow: 0 0 10px currentColor;
}

.phase-active {
    background: rgba(0, 217, 255, 0.15);
    border: 1px solid var(--color-primary);
    font-weight: 700;
    animation: phase-glow 2s ease-in-out infinite;
}

@keyframes phase-glow {
    0%, 100% { box-shadow: 0 0 10px rgba(0, 217, 255, 0.3); }
    50% { box-shadow: 0 0 20px rgba(0, 217, 255, 0.6); }
}

/* ========== DATA VISUALIZATION ========== */
.viz-container {
    background: rgba(10, 14, 39, 0.6);
    border-radius: 12px;
    padding: 24px;
    border: 1px solid var(--color-border);
}

/* ========== TELEMETRY SECTION ========== */
.telemetry-row {
    display: grid;
    grid-template-columns: 200px 1fr;
    gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid rgba(160, 179, 216, 0.1);
    align-items: center;
}

.telemetry-row:last-child {
    border-bottom: none;
}

.telemetry-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--color-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.telemetry-value {
    font-family: var(--font-display);
    font-size: 1.1rem;
    color: var(--color-primary);
    font-weight: 600;
}

.telemetry-bar {
    height: 8px;
    background: rgba(0, 217, 255, 0.2);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}

.telemetry-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--color-primary), var(--color-success));
    border-radius: 4px;
    transition: width 0.5s ease;
    box-shadow: 0 0 10px var(--color-primary);
}

/* ========== FOOTER ========== */
.footer-note {
    background: rgba(31, 40, 71, 0.3);
    border: 1px solid var(--color-border);
    border-radius: 8px;
    padding: 16px 24px;
    font-size: 0.8rem;
    color: var(--color-text-muted);
    text-align: center;
    margin-top: 32px;
}

/* ========== SCROLLBAR ========== */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: var(--color-bg-dark);
}

::-webkit-scrollbar-thumb {
    background: var(--color-bg-light);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--color-primary);
}

/* ========== RESPONSIVE ========== */
@media (max-width: 768px) {
    .main-title {
        font-size: 2.5rem;
    }
    
    .metric-grid {
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    }
    
    .phase-bar {
        flex-direction: column;
        align-items: flex-start;
    }
}

/* ========== LOADING ANIMATION ========== */
@keyframes slideInFromTop {
    0% {
        opacity: 0;
        transform: translateY(-30px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

.glass-panel {
    animation: slideInFromTop 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-panel:nth-child(2) { animation-delay: 0.1s; }
.glass-panel:nth-child(3) { animation-delay: 0.2s; }
.glass-panel:nth-child(4) { animation-delay: 0.3s; }

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(f"""
<div class="header-container">
  <div class="main-title">⬡ ASTRORISK</div>
  <div class="subtitle">Space Weather Operations Center</div>
  <div class="timestamp">LIVE TELEMETRY • {current_time}</div>
</div>
""", unsafe_allow_html=True)

# ================= CRITICAL ALERT =================
st.markdown("""
<div class="alert-critical">
  <div class="alert-title">⚠ SEVERE GEOMAGNETIC STORM DETECTED</div>
  <p class="alert-message">
    <strong>G3-class event in progress.</strong> Elevated radiation flux and atmospheric drag affecting orbital assets. 
    HF radio propagation severely degraded at high latitudes. Immediate operational mitigation protocols recommended for satellite operators.
  </p>
</div>
""", unsafe_allow_html=True)

# ================= OVERALL RISK GAUGE =================
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    # Main risk gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=78,
        delta={'reference': 45, 'position': "top"},
        number={
            'font': {'size': 72, 'color': '#00d9ff', 'family': 'Orbitron'},
            'suffix': '/100'
        },
        title={
            'text': "OVERALL RISK SCORE",
            'font': {'size': 18, 'color': '#a0b3d8', 'family': 'Orbitron'}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 2,
                'tickcolor': "#a0b3d8",
                'tickfont': {'color': '#a0b3d8', 'family': 'JetBrains Mono'}
            },
            'bar': {'color': "#ff1744", 'thickness': 0.8},
            'bgcolor': "rgba(31, 40, 71, 0.3)",
            'borderwidth': 2,
            'bordercolor': "rgba(0, 217, 255, 0.3)",
            'steps': [
                {'range': [0, 33], 'color': 'rgba(0, 245, 160, 0.3)'},
                {'range': [33, 66], 'color': 'rgba(255, 133, 0, 0.3)'},
                {'range': [66, 100], 'color': 'rgba(255, 23, 68, 0.3)'},
            ],
            'threshold': {
                'line': {'color': "#ffbe0b", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    
    gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#e8edf4", 'family': "JetBrains Mono"},
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    st.plotly_chart(gauge, use_container_width=True)

with col2:
    # Key metrics
    st.markdown("""
    <div style="padding: 20px 0;">
        <div class="metric-box">
            <div class="metric-label">Kp Index</div>
            <div class="metric-value">7<span class="metric-unit">/9</span></div>
        </div>
        <div style="height: 16px;"></div>
        <div class="metric-box">
            <div class="metric-label">Storm Level</div>
            <div class="metric-value" style="color: #ff1744;">G3</div>
        </div>
        <div style="height: 16px;"></div>
        <div class="metric-box">
            <div class="metric-label">Duration</div>
            <div class="metric-value" style="font-size: 1.4rem;">6<span class="metric-unit">h 24m</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ================= PHASE INDICATOR =================
st.markdown("""
<div class="phase-bar">
    <div class="phase-item">
        <div class="phase-indicator" style="background: #00f5a0;"></div>
        <span style="color: #a0b3d8;">Normal</span>
    </div>
    <div class="phase-item">
        <div class="phase-indicator" style="background: #ffbe0b;"></div>
        <span style="color: #a0b3d8;">Watch</span>
    </div>
    <div class="phase-item phase-active">
        <div class="phase-indicator" style="background: #ff1744;"></div>
        <span style="color: #ff1744;">STORM ACTIVE</span>
    </div>
    <div class="phase-item">
        <div class="phase-indicator" style="background: #6b7a99;"></div>
        <span style="color: #a0b3d8;">Recovery</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

# ================= SECTOR IMPACT CARDS =================
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown('<h2 style="font-family: Orbitron; margin-bottom: 24px;">⬢ SECTOR IMPACT ASSESSMENT</h2>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="status-card card-critical">
        <div class="card-title">🛰️ SATELLITE OPERATIONS</div>
        <div class="card-label">PRIMARY RISK</div>
        <div class="card-value">Solar energetic particle radiation + atmospheric drag increase</div>
        <div class="card-label">ROOT CAUSE</div>
        <div class="card-value">Proton flux >10 MeV exceeding 10³ pfu • Kp=7 causing thermospheric heating</div>
        <div class="card-action" style="background: rgba(255, 23, 68, 0.2); color: #ff1744; border: 1px solid #ff1744;">
            ⚠ ENTER SAFE MODE
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="status-card card-warning">
        <div class="card-title">✈️ AVIATION</div>
        <div class="card-label">PRIMARY RISK</div>
        <div class="card-value">HF communication blackouts on polar routes</div>
        <div class="card-label">ROOT CAUSE</div>
        <div class="card-value">D-layer absorption + ionospheric scintillation at high latitudes</div>
        <div class="card-action" style="background: rgba(255, 133, 0, 0.2); color: #ff8500; border: 1px solid #ff8500;">
            ⚡ REROUTE POLAR FLIGHTS
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="status-card card-safe">
        <div class="card-title">⚡ POWER GRID</div>
        <div class="card-label">PRIMARY RISK</div>
        <div class="card-value">Minimal geomagnetically induced current (GIC) exposure</div>
        <div class="card-label">ROOT CAUSE</div>
        <div class="card-value">Kp=7 remains below critical GIC threshold for most grid topologies</div>
        <div class="card-action" style="background: rgba(0, 245, 160, 0.2); color: #00f5a0; border: 1px solid #00f5a0;">
            ✓ MONITOR ONLY
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

# ================= SPACE WEATHER PARAMETERS =================
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown('<h2 style="font-family: Orbitron; margin-bottom: 24px;">⬢ REAL-TIME SPACE WEATHER PARAMETERS</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="viz-container">
        <div class="telemetry-row">
            <div class="telemetry-label">Solar Wind Speed</div>
            <div>
                <div class="telemetry-value">685 km/s</div>
                <div class="telemetry-bar">
                    <div class="telemetry-bar-fill" style="width: 82%;"></div>
                </div>
            </div>
        </div>
        <div class="telemetry-row">
            <div class="telemetry-label">IMF Bz Component</div>
            <div>
                <div class="telemetry-value" style="color: #ff1744;">-18.3 nT</div>
                <div class="telemetry-bar">
                    <div class="telemetry-bar-fill" style="width: 91%; background: linear-gradient(90deg, #ff1744, #ff006e);"></div>
                </div>
            </div>
        </div>
        <div class="telemetry-row">
            <div class="telemetry-label">Proton Flux (>10 MeV)</div>
            <div>
                <div class="telemetry-value">2,450 pfu</div>
                <div class="telemetry-bar">
                    <div class="telemetry-bar-fill" style="width: 75%;"></div>
                </div>
            </div>
        </div>
        <div class="telemetry-row">
            <div class="telemetry-label">Electron Flux (>2 MeV)</div>
            <div>
                <div class="telemetry-value">1.8e5 e⁻/cm²/s</div>
                <div class="telemetry-bar">
                    <div class="telemetry-bar-fill" style="width: 68%;"></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="viz-container">
        <div class="telemetry-row">
            <div class="telemetry-label">Dst Index</div>
            <div>
                <div class="telemetry-value" style="color: #ff8500;">-142 nT</div>
                <div class="telemetry-bar">
                    <div class="telemetry-bar-fill" style="width: 85%; background: linear-gradient(90deg, #ff8500, #ffbe0b);"></div>
                </div>
            </div>
        </div>
        <div class="telemetry-row">
            <div class="telemetry-label">F10.7 Solar Flux</div>
            <div>
                <div class="telemetry-value">188 sfu</div>
                <div class="telemetry-bar">
                    <div class="telemetry-bar-fill" style="width: 72%;"></div>
                </div>
            </div>
        </div>
        <div class="telemetry-row">
            <div class="telemetry-label">Sunspot Number</div>
            <div>
                <div class="telemetry-value">124</div>
                <div class="telemetry-bar">
                    <div class="telemetry-bar-fill" style="width: 65%;"></div>
                </div>
            </div>
        </div>
        <div class="telemetry-row">
            <div class="telemetry-label">X-Ray Background</div>
            <div>
                <div class="telemetry-value">C2.4</div>
                <div class="telemetry-bar">
                    <div class="telemetry-bar-fill" style="width: 45%;"></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

# ================= TIME SERIES VISUALIZATION =================
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown('<h2 style="font-family: Orbitron; margin-bottom: 24px;">⬢ 48-HOUR TREND ANALYSIS</h2>', unsafe_allow_html=True)

# Generate sample time series data
hours = pd.date_range(end=datetime.now(), periods=48, freq='h')
kp_values = np.concatenate([
    np.random.uniform(2, 4, 24),
    np.random.uniform(5, 7, 16),
    np.random.uniform(3, 5, 8)
])

solar_wind = np.concatenate([
    np.random.uniform(350, 450, 24),
    np.random.uniform(550, 720, 16),
    np.random.uniform(400, 500, 8)
])

# Create interactive time series
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=hours,
    y=kp_values,
    name='Kp Index',
    line=dict(color='#00d9ff', width=3),
    fill='tozeroy',
    fillcolor='rgba(0, 217, 255, 0.2)',
    hovertemplate='<b>Kp Index</b><br>Time: %{x}<br>Value: %{y:.1f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=hours,
    y=solar_wind / 100,
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

st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

# ================= EXPLANATION SECTION =================
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown('<h2 style="font-family: Orbitron; margin-bottom: 24px;">⬢ STORM GENESIS & DYNAMICS</h2>', unsafe_allow_html=True)

st.markdown("""
<div style="color: #a0b3d8; line-height: 1.8; font-size: 0.95rem;">
<p style="margin-bottom: 16px;">
The current G3-class geomagnetic storm was initiated by a coronal mass ejection (CME) that departed the Sun 
approximately 72 hours ago. The plasma cloud's arrival at Earth triggered the following cascade:
</p>

<ul style="margin-left: 20px; margin-bottom: 16px;">
<li style="margin-bottom: 12px;"><strong style="color: #00d9ff;">Magnetopause Compression:</strong> 
Solar wind dynamic pressure compressed Earth's magnetosphere from ~10 RE to ~7 RE, exposing 
geostationary satellites to elevated particle flux.</li>

<li style="margin-bottom: 12px;"><strong style="color: #00d9ff;">Magnetic Reconnection:</strong> 
Sustained southward IMF Bz (-18.3 nT) enabled efficient magnetic reconnection at the dayside magnetopause, 
injecting solar wind energy into the magnetosphere.</li>

<li style="margin-bottom: 12px;"><strong style="color: #00d9ff;">Ring Current Enhancement:</strong> 
Energized particles were transported to the ring current, producing the observed Dst depression of -142 nT 
and elevating the Kp index to 7.</li>

<li style="margin-bottom: 12px;"><strong style="color: #00d9ff;">Ionospheric Response:</strong> 
Auroral electrojet currents induced ionospheric irregularities, causing HF radio absorption and 
GPS scintillation at high latitudes.</li>
</ul>

<p style="color: #ffbe0b; font-weight: 600; margin-top: 20px;">
⚡ Recovery Phase: Storm conditions expected to persist for 8-12 hours before gradual subsidence as 
solar wind speed decreases and IMF Bz rotates northward.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

# ================= DETAILED TELEMETRY EXPANDER =================
with st.expander("📡 View Detailed Telemetry & Forecast Models", expanded=False):
    tab1, tab2, tab3 = st.tabs(["Solar Wind", "Magnetosphere", "Ionosphere"])
    
    with tab1:
        st.markdown("#### Solar Wind Parameters (ACE/DSCOVR L1)")
        st.markdown("""
        - **Bulk Velocity:** 685 km/s (↑ from 420 km/s baseline)
        - **Proton Density:** 12.4 cm⁻³
        - **Dynamic Pressure:** 8.2 nPa
        - **IMF Magnitude:** 22.6 nT
        - **IMF Bz:** -18.3 nT (southward, geoeffective)
        - **Alfvén Speed:** 142 km/s
        - **Plasma Beta:** 1.8
        """)
    
    with tab2:
        st.markdown("#### Magnetospheric Indices")
        st.markdown("""
        - **Kp Index:** 7 (G3 storm)
        - **Dst Index:** -142 nT (moderate storm)
        - **AE Index:** 1,245 nT (strong auroral electrojet)
        - **Magnetopause Standoff:** 7.2 RE
        - **Plasmapause Location:** L=3.8
        """)
    
    with tab3:
        st.markdown("#### Ionospheric Conditions")
        st.markdown("""
        - **TEC (Total Electron Content):** 45 TECU (elevated)
        - **foF2 (Critical Frequency):** 8.2 MHz
        - **Scintillation Index (S4):** 0.68 at high latitudes
        - **HF Absorption:** 6-8 dB at polar cap
        - **Auroral Oval Latitude:** 58°N geomagnetic
        """)

# ================= FOOTER =================
st.markdown("""
<div class="footer-note">
    <strong>ASTRORISK</strong> is an operational decision-support system. All advisories require human validation 
    before implementation. Data sources: NOAA SWPC, NASA ACE/DSCOVR, USGS Geomagnetism Program. 
    <br><br>
    <span style="color: #6b7a99;">Classification: OPERATIONAL // Distribution: AUTHORIZED PERSONNEL ONLY</span>
</div>
""", unsafe_allow_html=True)