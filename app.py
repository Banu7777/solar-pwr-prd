import streamlit as st
import torch
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

from model import NHiTS

# ==============================
# PAGE + STYLE
# ==============================
st.set_page_config(page_title="PV Gücü Proqnozu (AI)", layout="wide")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown(
    "<h2 style='margin-bottom:0'>☀️ Real vaxt meteoroloji məlumatları ilə PV gücü (kW) proqnozu</h2>"
    "<div style='color:gray'>Open‑Meteo + N‑HiTS (2–3 saat üfüq)</div>",
    unsafe_allow_html=True
)

# ==============================
# CONSTANTS
# ==============================
SEQ_LEN = 168  # 7 days (hourly)
ETA_BASE = 0.85
TEMP_COEFF = 0.004  # ~0.4% per +1°C above 25°C

FEATURES = [
    "temperature", "cloudcover",
    "shortwave_radiation",
    "hour_sin", "hour_cos",
    "G_lag1", "G_lag3", "G_lag6", "G_lag12", "G_lag24",
    "pv_lag1", "pv_lag3", "pv_lag6", "pv_lag12", "pv_lag24",
    "pv_roll6_mean", "pv_roll12_mean", "pv_roll24_mean",
    "pv_roll6_std", "pv_roll12_std", "pv_roll24_std",
]

AZ_CITIES = {
    "Bakı": (40.4093, 49.8671),
    "Gəncə": (40.6828, 46.3606),
    "Sumqayıt": (40.5897, 49.6686),
    "Mingəçevir": (40.7703, 47.0496),
    "Şəki": (41.1919, 47.1706),
    "Lənkəran": (38.7543, 48.8511),
    "Naxçıvan": (39.2089, 45.4122),
    "Quba": (41.3611, 48.5139),
    "Şamaxı": (40.6314, 48.6414),
    "Xüsusi koordinat": None
}

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("⚙️ Parametrlər")

city = st.sidebar.selectbox("Şəhər seçin", list(AZ_CITIES.keys()), index=0)
if city != "Xüsusi koordinat":
    lat, lon = AZ_CITIES[city]
    st.sidebar.caption(f"Seçilmiş şəhər: **{city}**  |  Koordinatlar: {lat:.4f}, {lon:.4f}")
else:
    lat = st.sidebar.number_input("Latitude", value=40.4093, format="%.4f")
    lon = st.sidebar.number_input("Longitude", value=49.8671, format="%.4f")

horizon = st.sidebar.selectbox("Proqnoz üfüqü (saat)", [2, 3], index=1)
p_rated = st.sidebar.number_input("PV sistemin nominal gücü (kW)", min_value=1.0, max_value=50.0, value=5.0, step=1.0)

st.sidebar.divider()
st.sidebar.caption("Qeyd: PV gücü real stansiya ölçməsi deyil, radiasiya və temperaturdan hesablanan proxy dəyərdir.")

run_btn = st.sidebar.button("🔮 Proqnozu hesabla", use_container_width=True)

MODEL_PATH = f"n_hits_solar_pv_model_h{horizon}.pth"

# ==============================
# LOAD SCALER + MODEL
# ==============================
@st.cache_resource(show_spinner=False)
def load_model_and_scaler(model_path: str):
    mean = np.load("solar_scaler_mean.npy")
    scale = np.load("solar_scaler_scale.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NHiTS(seq_len=SEQ_LEN, num_features=len(FEATURES), hidden_size=256, num_blocks=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device, mean, scale

def std_scale(df: pd.DataFrame, mean: np.ndarray, scale: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df[FEATURES] = (df[FEATURES] - mean) / scale
    return df

def pv_proxy_kw(G_wm2: np.ndarray, temp_c: np.ndarray, p_rated_kw: float) -> np.ndarray:
    G = np.maximum(G_wm2, 0.0)
    eta_temp = 1 - TEMP_COEFF * (temp_c - 25.0)
    eta_temp = np.clip(eta_temp, 0.70, 1.05)
    return np.maximum(p_rated_kw * (G / 1000.0) * ETA_BASE * eta_temp, 0.0)

def add_features(df: pd.DataFrame, p_rated_kw: float) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df["pv_power_kw"] = pv_proxy_kw(
        df["shortwave_radiation"].values.astype(float),
        df["temperature"].values.astype(float),
        p_rated_kw
    )

    df["hour"] = df["time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    for k in [1, 3, 6, 12, 24]:
        df[f"G_lag{k}"] = df["shortwave_radiation"].shift(k)
        df[f"pv_lag{k}"] = df["pv_power_kw"].shift(k)

    df["pv_roll6_mean"] = df["pv_power_kw"].rolling(6).mean()
    df["pv_roll12_mean"] = df["pv_power_kw"].rolling(12).mean()
    df["pv_roll24_mean"] = df["pv_power_kw"].rolling(24).mean()

    df["pv_roll6_std"] = df["pv_power_kw"].rolling(6).std()
    df["pv_roll12_std"] = df["pv_power_kw"].rolling(12).std()
    df["pv_roll24_std"] = df["pv_power_kw"].rolling(24).std()

    df = df.dropna().reset_index(drop=True)
    return df

@st.cache_data(ttl=900, show_spinner=False)
def fetch_recent_hours(lat: float, lon: float, hours: int = 320) -> pd.DataFrame:
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&hourly=shortwave_radiation,temperature_2m,cloudcover"
        "&past_days=10&forecast_days=2&timezone=auto"
    )
    js = requests.get(url, timeout=30).json()
    if "hourly" not in js:
        raise RuntimeError("Open‑Meteo cavabında hourly hissəsi tapılmadı.")
    df = pd.DataFrame({
        "time": js["hourly"]["time"],
        "shortwave_radiation": js["hourly"]["shortwave_radiation"],
        "temperature": js["hourly"]["temperature_2m"],
        "cloudcover": js["hourly"]["cloudcover"],
    })
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    if len(df) > hours:
        df = df.iloc[-hours:].reset_index(drop=True)
    return df

def predict_pv_kw(model, device, df_scaled: pd.DataFrame) -> float:
    x = df_scaled[FEATURES].values[-SEQ_LEN:]
    x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        yhat = model(x).item()
    return float(yhat)

# ==============================
# UX: instructions until click
# ==============================
if not run_btn:
    st.info(
        "Soldakı paneldən şəhəri, proqnoz üfüqünü (2 və ya 3 saat) və PV gücünü seçin, sonra **‘Proqnozu hesabla’** düyməsinə basın."
    )
    st.stop()

# ==============================
# MAIN FLOW
# ==============================
try:
    model, device, mean, scale = load_model_and_scaler(MODEL_PATH)
except FileNotFoundError as e:
    st.error(
        f"Model və ya scaler faylı tapılmadı: {e}\n\n"
        "Bu faylların eyni qovluqda olduğuna əmin olun:\n"
        "- solar_scaler_mean.npy\n- solar_scaler_scale.npy\n"
        f"- {MODEL_PATH}"
    )
    st.stop()

try:
    df_raw = fetch_recent_hours(lat, lon, hours=320)
except Exception as e:
    st.error(f"Open‑Meteo məlumatını çəkmək alınmadı: {e}")
    st.stop()

df_feat = add_features(df_raw, p_rated)
df_scaled = std_scale(df_feat, mean, scale)
pred_kw = predict_pv_kw(model, device, df_scaled)

# ==============================
# PHYSICAL GATING (PV cannot be negative, night-time handling)
# ==============================
current_radiation = df_raw["shortwave_radiation"].iloc[-1]

# If it is night or very low radiation, PV power must be zero
if current_radiation < 5:   # W/m² threshold
    pred_kw = 0.0
else:
    # Just in case model outputs a small negative value
    pred_kw = max(0.0, pred_kw)


now_time = df_feat["time"].iloc[-1]
future_time = now_time + timedelta(hours=int(horizon))

# ==============================
# TOP METRICS
# ==============================
st.caption(f"Məkan: {city}  •  Proqnoz üfüqü: +{horizon} saat  •  Yüklənən model: {MODEL_PATH}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("İndiki radiasiya", f"{df_raw['shortwave_radiation'].iloc[-1]:.0f} W/m²")
col2.metric("İndiki temperatur", f"{df_raw['temperature'].iloc[-1]:.1f} °C")
col3.metric("İndiki buludluluq", f"{df_raw['cloudcover'].iloc[-1]:.0f} %")
col4.metric(f"PV gücü proqnozu (+{horizon}h)", f"{pred_kw:.2f} kW")

st.caption(f"Proqnoz vaxtı: {future_time.strftime('%Y-%m-%d %H:%M')} (local timezone)")

# ==============================
# VISUAL 1
# ==============================
st.markdown(f"### 1) Son 72 saat (tarixi) PV gücü (proxy) + +{horizon} saat proqnoz nöqtəsi")
last72 = df_feat.iloc[-72:].copy()

fig1, ax1 = plt.subplots(figsize=(12, 3.6))
ax1.fill_between(last72["time"], last72["pv_power_kw"], alpha=0.35)
ax1.plot(last72["time"], last72["pv_power_kw"], linewidth=1.2)
ax1.scatter([future_time], [pred_kw], s=65, marker="o")
ax1.text(future_time, pred_kw, f"  +{horizon}h", va="center")

ax1.set_ylabel("PV gücü (kW)")
ax1.set_xlabel("Zaman")
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# ==============================
# VISUAL 2
# ==============================
st.markdown("### 2) Radiasiya–PV gücü əlaqəsi (son 72 saat)")
fig2, ax2 = plt.subplots(figsize=(6.8, 4.2))
ax2.scatter(last72["shortwave_radiation"], last72["pv_power_kw"], alpha=0.6)
ax2.set_xlabel("Günəş radiasiyası (W/m²)")
ax2.set_ylabel("PV gücü (kW)")
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

# ==============================
# VISUAL 3
# ==============================
st.markdown("### 3) Günlük PV profili (son 7 gün: saatlara görə orta PV gücü)")
last7d = df_feat.iloc[-24*7:].copy()
profile = last7d.groupby(last7d["time"].dt.hour)["pv_power_kw"].mean().reindex(range(24), fill_value=0)

fig3, ax3 = plt.subplots(figsize=(10, 3.4))
ax3.plot(profile.index, profile.values, linewidth=2.0)
ax3.set_xticks(range(0, 24, 2))
ax3.set_xlabel("Saat (0–23)")
ax3.set_ylabel("Orta PV gücü (kW)")
ax3.grid(True, alpha=0.3)
st.pyplot(fig3)

with st.expander("ℹ️ Qısa izah (münsif üçün)"):
    st.markdown(
        "- Qrafik 1 və 2-də **son 72 saatın tarixi məlumatları** göstərilir və seçilən üfüqə görə (+2/+3 saat) proqnoz nöqtəsi əlavə olunur.\n"
        "- Qrafik 3-də isə **günəş enerjisinə xas gündəlik profil** (son 7 günün ortalaması) göstərilir; bu, layihəni külək proqnozu layihəsindən vizual olaraq fərqləndirir."
    )
