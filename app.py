import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy import units as u
from astropy.time import Time, TimeDelta

# --------------------- Generate Telemetry Data --------------------- #
time = np.arange(0, 500, 1)

temperature = 25 + 2*np.sin(0.02*time)
voltage = 5 + 0.2*np.sin(0.01*time)
current = 2 + 0.1*np.sin(0.03*time)
signal_strength = 90 + 5*np.sin(0.02*time)

# Introduce anomalies
temperature[200:220] += 15
voltage[350:360] -= 2

telemetry = pd.DataFrame({
    "time": time,
    "temperature": temperature,
    "voltage": voltage,
    "current": current,
    "signal_strength": signal_strength
})

# --------------------- Anomaly Detection --------------------- #
features = telemetry[["temperature","voltage","current","signal_strength"]]
model = IsolationForest(contamination=0.05, random_state=42)
telemetry["anomaly"] = model.fit_predict(features)   # -1 = anomaly

# --------------------- Orbit & Space Traffic --------------------- #
epoch = Time("2025-01-01 00:00:00", scale="utc")
sat1 = Orbit.circular(Earth, alt=500 * u.km, epoch=epoch)
sat2 = Orbit.circular(Earth, alt=505 * u.km, epoch=epoch)

# Sample times along orbit (100 points)
times = TimeDelta(np.linspace(0, sat1.period.to(u.s).value, 100) * u.s)

# Satellite positions along orbit
sat1_coords = np.array([sat1.propagate(t).r.to(u.km).value for t in times])
sat2_coords = np.array([sat2.propagate(t).r.to(u.km).value for t in times])

# Calculate minimum distance along orbit for collision risk
distances = np.linalg.norm(sat1_coords - sat2_coords, axis=1)
min_distance = np.min(distances)

risk_status = "SAFE 🟢"
if min_distance < 50:
    risk_status = "WARNING 🟡"
if min_distance < 10:
    risk_status = "CRITICAL 🔴"

# --------------------- Streamlit App --------------------- #
st.set_page_config(page_title="TelemetriX", page_icon="🛰️", layout="wide")
st.title("🛰️ TelemetriX – Satellite Health & Space Traffic Monitoring")

# Telemetry Line Chart
fig1 = px.line(
    telemetry, 
    x="time", 
    y=["temperature","voltage","current","signal_strength"],
    title="Telemetry Signals"
)
st.plotly_chart(fig1, use_container_width=True)

# Detected Anomalies
st.subheader("🚨 Detected Anomalies")
st.dataframe(telemetry[telemetry["anomaly"] == -1])

# Space Traffic Management
st.subheader("🌐 Space Traffic Management (STM)")
st.write(f"Minimum Satellite distance along orbit: **{min_distance:.2f} km**")
st.write(f"Risk Status: **{risk_status}**")

# --------------------- Interactive 3D Orbit Plot --------------------- #
fig2 = go.Figure()

# Earth
u_sphere, v_sphere = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
x = 6371 * np.cos(u_sphere) * np.sin(v_sphere)
y = 6371 * np.sin(u_sphere) * np.sin(v_sphere)
z = 6371 * np.cos(v_sphere)
fig2.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Blues', opacity=0.6, name='Earth'))

# Satellite 1 orbit line + current position
fig2.add_trace(go.Scatter3d(
    x=sat1_coords[:,0],
    y=sat1_coords[:,1],
    z=sat1_coords[:,2],
    mode='lines+markers',
    name='Satellite 1 Orbit',
    line=dict(color='red'),
    marker=dict(size=3)
))

# Satellite 2 orbit line + current position
fig2.add_trace(go.Scatter3d(
    x=sat2_coords[:,0],
    y=sat2_coords[:,1],
    z=sat2_coords[:,2],
    mode='lines+markers',
    name='Satellite 2 Orbit',
    line=dict(color='green'),
    marker=dict(size=3)
))

fig2.update_layout(scene=dict(
    xaxis_title='X (km)',
    yaxis_title='Y (km)',
    zaxis_title='Z (km)',
    aspectmode='data'
), title="3D Orbit Visualization")

st.plotly_chart(fig2, use_container_width=True)
