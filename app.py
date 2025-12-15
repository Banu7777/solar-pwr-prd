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
    "<h2 style='margin-bottom:0'>☀️ Günəş enerjisi istehsalının qısamüddətli AI proqnozu (PV gücü, kW)</h2>"
    "<div style='color:gray'>Real vaxt meteoroloji məlumatları + N-HiTS modeli | +2 / +3 saat</div>",
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
    "pv_roll6_std", "pv_roll12_std", "pv_roll24_std"
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

with st.expander("📘 Layihə haqqında ümumi məlumat", expanded=True):
    st.markdown(
        """
Bu veb-tətbiq günəş panellərinin **elektrik enerjisi istehsalını (PV gücü, kW)** 
qısamüddətli perspektivdə proqnozlaşdırmaq üçün hazırlanmışdır. 
Proqnozlar real vaxtda əldə olunan **meteoroloji məlumatlar** 
(günəş radiasiyası, temperatur və buludluluq) əsasında hesablanır.

Layihənin əsas məqsədi günəş enerjisi sistemlərində 
**istehsalın əvvəlcədən qiymətləndirilməsi**, 
enerji planlaşdırılması və şəbəkə balansının yaxşılaşdırılmasına töhfə verməkdir.
Bu məqsədlə zaman sırası məlumatları üçün uyğun olan **N-HiTS dərin öyrənmə modeli** istifadə edilmişdir.

Model son **7 günün saatlıq məlumatlarını** analiz edərək 
günəş enerjisi istehsalını **+2 və ya +3 saat** sonrakı vaxt üçün proqnozlaşdırır.
Alınan nəticələr fiziki məhdudiyyətlər nəzərə alınmaqla (gecə saatlarında istehsalın sıfır olması) təqdim olunur.
        """
    )

    st.markdown(
        "**İstifadə olunan əsas komponentlər:**\n"
        "- Məlumat mənbəyi: Open-Meteo (real vaxt meteoroloji API)\n"
        "- Giriş parametrləri: günəş radiasiyası, temperatur, buludluluq və zaman xüsusiyyətləri\n"
        "- Model: N-HiTS (Neural Hierarchical Interpolation for Time Series)\n"
        "- Çıxış: PV gücü proqnozu (kW)\n"
    )


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

def predict_pv_kw(model, device, df_scaled: pd.DataFrame, mean: np.ndarray, scale: np.ndarray) -> float:
    x = df_scaled[FEATURES].values[-SEQ_LEN:]
    x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        yhat_scaled = model(x).item()

    # ---- inverse transform for target if scaler includes it ----
    if len(mean) == len(FEATURES) + 1:
        yhat = yhat_scaled * float(scale[-1]) + float(mean[-1])
    else:
        # If target scale is not available, return raw (we'll clamp later)
        yhat = yhat_scaled

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
pred_kw = predict_pv_kw(model, device, df_scaled, mean, scale)

#------------------------------------------------------
# ==============================
# TIME + METRICS TIME FIX (NOW) + PHYSICAL GATING
# ==============================

# # 1) Pick "now_time" correctly (latest available hour <= current local clock time)
# tz = df_raw["time"].dt.tz
# now_clock = pd.Timestamp.now(tz=tz).floor("H") if tz is not None else pd.Timestamp.now().floor("H")
# now_time = df_raw.loc[df_raw["time"] <= now_clock, "time"].iloc[-1]

# # 2) Current (real) values at now_time (for metrics)
# current_radiation = float(df_raw.loc[df_raw["time"] == now_time, "shortwave_radiation"].iloc[0])
# current_temp = float(df_raw.loc[df_raw["time"] == now_time, "temperature"].iloc[0])
# current_cloud = float(df_raw.loc[df_raw["time"] == now_time, "cloudcover"].iloc[0])

# # 3) Future time for prediction display (+2/+3 hours from now_time)
# future_time = now_time + timedelta(hours=int(horizon))

# # 4) Physical gating should use radiation at FUTURE time (because you predict future PV)
# future_rad = df_raw.loc[df_raw["time"] == future_time, "shortwave_radiation"]
# if len(future_rad) > 0 and float(future_rad.iloc[0]) < 5:
#     pred_kw = 0.0
# else:
#     pred_kw = max(0.0, pred_kw)


from zoneinfo import ZoneInfo

# ==============================
# TIME (Baku) + DEBUG + PHYSICAL GATING (FIX)
# ==============================

# Make a Baku "clock" (even if df_raw time is naive)
now_clock_baku = pd.Timestamp.now(tz=ZoneInfo("Asia/Baku")).tz_localize(None).floor("H")

# Choose now_time as latest available hour <= Baku clock
now_time = df_raw.loc[df_raw["time"] <= now_clock_baku, "time"].iloc[-1]

# Future time
future_time = now_time + timedelta(hours=int(horizon))

# Radiation now / future
current_radiation = float(df_raw.loc[df_raw["time"] == now_time, "shortwave_radiation"].iloc[0])

future_rad_series = df_raw.loc[df_raw["time"] == future_time, "shortwave_radiation"]
future_radiation = float(future_rad_series.iloc[0]) if len(future_rad_series) > 0 else None

current_temp = float(df_raw.loc[df_raw["time"] == now_time, "temperature"].iloc[0])
current_cloud = float(df_raw.loc[df_raw["time"] == now_time, "cloudcover"].iloc[0])

# pred_kw = float(np.clip(pred_kw, 0.0, p_rated))

# ---- DEBUG (very important now) ----
with st.expander("🛠 Debug (time + radiation + raw prediction)", expanded=False):
    st.write("now_clock_baku:", now_clock_baku)
    st.write("now_time picked:", now_time)
    st.write("future_time:", future_time)
    st.write("current_radiation:", current_radiation)
    st.write("future_radiation:", future_radiation)
    st.write("pred_kw BEFORE gating:", pred_kw)

# Physical gating should use future radiation (since you predict +2/+3h)
if future_radiation is not None and future_radiation < 5:
    pred_kw = 0.0
else:
    pred_kw = max(0.0, pred_kw)


#------------------------------------------------------

# ==============================
# TOP METRICS
# ==============================
st.caption(f"Məkan: {city}  •  Proqnoz üfüqü: +{horizon} saat  •  Yüklənən model: {MODEL_PATH}")

col1, col2, col3, col4 = st.columns(4)

col1.metric("İndiki radiasiya", f"{current_radiation:.0f} W/m²")
col2.metric("İndiki temperatur", f"{current_temp:.1f} °C")
col3.metric("İndiki buludluluq", f"{current_cloud:.0f} %")
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

with st.expander("ℹ️ Qrafiklər haqqında izahlar"):
    st.markdown(
        "- Qrafik 1 və 2-də **son 72 saatın tarixi məlumatları** göstərilir və seçilən üfüqə görə (+2/+3 saat) proqnoz nöqtəsi əlavə olunur.\n"
        "- Qrafik 3-də isə **günəş enerjisinə xas gündəlik profil** (son 7 günün ortalaması) göstərilir; bu, layihəni külək proqnozu layihəsindən vizual olaraq fərqləndirir."
    )

import os

st.markdown("---")
st.markdown("## Əlavə analiz qrafikləri")

with st.expander("📌 Modelin dəqiqliyi və dəyişənlərin əhəmiyyətliliyi", expanded=False):
    colA, colB = st.columns(2)

    # 1) Accuracy plot (h2 vs h3)
    with colA:
            st.image("accuracy_h2_h3.png", use_container_width=True)
            st.caption(
                "Bu qrafik +2 və +3 saat üfüqləri üçün MAE və RMSE xətalarını müqayisə edir. "
                "Üfüq uzandıqca (3 saat) qeyri-müəyyənlik artdığı üçün xəta da adətən yüksəlir."
            )
    

    # 2) Feature importance plot (h3)
    with colB:
            st.image("feature_importance_h3.png", use_container_width=True)
            st.caption(
                "Bu qrafik permutation importance əsasında model üçün ən təsirli top-15 feature-i göstərir. "
                "Dəyərlər həmin feature qarışdırıldıqda MAE-nin nə qədər artdığını ifadə edir (artım böyükdürsə, feature daha vacibdir)."
            )
     

