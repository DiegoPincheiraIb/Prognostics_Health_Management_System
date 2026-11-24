import os

import httpx
import pandas as pd
import streamlit as st

from rul.data import COLUMNS

API = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="Industrial RUL", layout="wide")
st.title("Industrial RUL — demostración")
st.warning("Prototipo educativo: no utilizar para decisiones de mantenimiento o seguridad.")

try:
    info = httpx.get(f"{API}/model", timeout=30)
    info.raise_for_status()
    model = info.json()
except httpx.HTTPError as exc:
    st.error(f"API o modelo no disponible: {exc}")
    st.stop()

st.caption(f"Modelo: {model['name']} v{model['version']} · umbral: {model['threshold']} ciclos")
uploaded = st.file_uploader("Cargar un archivo CSV con columnas C-MAPSS", type="csv")
if uploaded:
    try:
        frame = pd.read_csv(uploaded)
        if list(frame.columns) != COLUMNS:
            raise ValueError(f"Se requieren las columnas, en orden: {', '.join(COLUMNS)}")
        available = frame.unit.unique()
        selected = st.selectbox("Motor en archivo", available)
        group = frame[frame.unit == selected].sort_values("cycle")
        readings = [{"cycle": int(row["cycle"]),
                     "settings": [float(row[f"setting_{i}"]) for i in range(1, 4)],
                     "sensors": [float(row[f"sensor_{i}"]) for i in range(1, 22)]}
                    for _, row in group.iterrows()]
        if st.button("Predecir"):
            result = httpx.post(f"{API}/predict", json={"unit": str(selected), "readings": readings},
                                timeout=60)
            result.raise_for_status()
            st.success(f"RUL: {result.json()['rul']:.1f} ciclos")
            if result.json()["alert"]:
                st.warning("Alerta: RUL en o bajo el umbral.")
    except (ValueError, KeyError, httpx.HTTPError) as exc:
        st.error(f"No se pudo procesar la solicitud: {exc}")

try:
    response = httpx.get(f"{API}/units", timeout=10)
    response.raise_for_status()
    units = response.json()
except httpx.HTTPError as exc:
    st.error(f"No se pudo recuperar la lista de motores: {exc}")
    st.stop()

if not units:
    st.info("Aún no existen predicciones guardadas.")
else:
    unit = st.selectbox("Motor guardado", units)
    try:
        response = httpx.get(f"{API}/history/{unit}", timeout=10)
        response.raise_for_status()
        history = response.json()
        if history:
            predictions = pd.DataFrame(history).sort_values("created_at")
            st.subheader("Evolución de predicciones")
            st.line_chart(predictions.set_index("created_at")[["rul", "threshold"]])
            latest = history[0]
            st.metric("Último RUL", f"{latest['rul']:.1f} ciclos",
                      help=f"Modelo v{latest['model_version']}")
            if latest["alert"]:
                st.warning("Alerta activa para la última predicción.")
            readings = pd.DataFrame(latest["readings"])
            sensor = st.selectbox("Sensor", range(1, 22))
            st.subheader("Evolución del sensor en la última predicción")
            st.line_chart(pd.DataFrame({"cycle": readings["cycle"],
                                        "value": readings["sensors"].map(lambda x: x[sensor - 1])})
                          .set_index("cycle"))
    except httpx.HTTPError as exc:
        st.error(f"No se pudo recuperar el historial: {exc}")