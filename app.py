import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time

# 1. Configuración de página
st.set_page_config(
    page_title="Predicción de Tráfico Cajamarca 2026",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .status-badge-green {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .status-badge-yellow {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .status-badge-red {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# 2. Cargar modelo serializado
@st.cache_resource
def load_bundle():
    try:
        bundle = joblib.load("modelo_xgboost_cajamarca.pkl")
        return bundle
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

bundle = load_bundle()

# Encabezado principal
st.markdown('<div class="main-header">🚦 Sistema Predictivo de Tráfico Vehicular — Cajamarca 2026</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Modelo de Machine Learning (XGBoost) para Semaforización Adaptativa y Reducción de la Congestión Urbana</div>', unsafe_allow_html=True)

if bundle is None:
    st.warning("No se encontró el archivo del modelo `modelo_xgboost_cajamarca.pkl`. Asegúrese de que esté en el mismo directorio.")
    st.stop()

modelo = bundle["modelo"]
features = bundle["features"]
cols_via = bundle["cols_via"]
umbrales = bundle["umbrales"]
composicion_modal = bundle["composicion_modal"]

# 3. Barra Lateral: Parámetros Operativos
st.sidebar.header("⚙️ Parámetros de Simulación")

corredor_sel = st.sidebar.selectbox(
    "Corredor Vial Estructurante",
    ["Av. Hoyos Rubio (Evitamiento Norte)", "Av. Atahualpa (Evitamiento Sur)", "Av. Los Héroes"],
    index=0
)

# Mapeo de nombre
via_map = {
    "Av. Hoyos Rubio (Evitamiento Norte)": "Av. Hoyos Rubio",
    "Av. Atahualpa (Evitamiento Sur)": "Av. Atahualpa",
    "Av. Los Héroes": "Av. Los Héroes"
}
via_clean = via_map[corredor_sel]
umb_via = umbrales[via_clean]

st.sidebar.markdown("---")
st.sidebar.subheader("🕒 Contexto Temporal")

dias_dict = {
    "Lunes": 0, "Martes": 1, "Miércoles": 2, "Jueves": 3,
    "Viernes": 4, "Sábado": 5, "Domingo": 6
}
dia_nom = st.sidebar.selectbox("Día de la Semana", list(dias_dict.keys()), index=0)
dia_semana = dias_dict[dia_nom]

hora_sel = st.sidebar.slider(
    "Hora del Día (Formato decimal)",
    min_value=6.5,
    max_value=20.0,
    value=7.75,
    step=0.25,
    format="%.2f h"
)

# Convertir hora decimal a texto legible
hora_int = int(hora_sel)
min_int = int(round((hora_sel - hora_int) * 60))
hora_txt = f"{hora_int:02d}:{min_int:02d}"
st.sidebar.caption(f"⏰ Hora seleccionada: **{hora_txt}**")

es_feriado_bool = st.sidebar.checkbox("¿Es Feriado o Día No Laborable?", value=False)
feriado = 1 if es_feriado_bool else 0

st.sidebar.markdown("---")
st.sidebar.subheader("🌧️ Condiciones Meteorológicas (Open-Meteo)")

clima_preset = st.sidebar.radio(
    "Preset Climático Andino:",
    ["Seco / Despejado (18°C, 0 mm)", "Llovizna Ligera (14°C, 1.2 mm)", "Lluvia Intensa (11°C, 5.8 mm)", "Manual"]
)

if clima_preset == "Seco / Despejado (18°C, 0 mm)":
    temperatura = 18.0
    precipitacion = 0.0
elif clima_preset == "Llovizna Ligera (14°C, 1.2 mm)":
    temperatura = 14.0
    precipitacion = 1.2
elif clima_preset == "Lluvia Intensa (11°C, 5.8 mm)":
    temperatura = 11.0
    precipitacion = 5.8
else:
    temperatura = st.sidebar.slider("Temperatura (°C)", 8.0, 26.0, 16.0, 0.5)
    precipitacion = st.sidebar.slider("Precipitación acumulada (mm)", 0.0, 15.0, 0.0, 0.2)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Inercia del Flujo Vehicular")

# Estimación base según vía y hora
if 7.25 <= hora_sel <= 8.75 or 12.75 <= hora_sel <= 14.0 or 18.0 <= hora_sel <= 19.5:
    flujo_base = umb_via["P85"] * 0.95
else:
    flujo_base = umb_via["P50"] * 0.85

lag_1 = st.sidebar.number_input("Flujo hace 15 min (vehículos)", min_value=50, max_value=1200, value=int(flujo_base))
lag_2 = st.sidebar.number_input("Flujo hace 30 min (vehículos)", min_value=50, max_value=1200, value=int(flujo_base * 0.92))
lag_4 = st.sidebar.number_input("Flujo hace 60 min (vehículos)", min_value=50, max_value=1200, value=int(flujo_base * 0.88))
lag_dia = st.sidebar.number_input("Mismo intervalo ayer (vehículos)", min_value=50, max_value=1200, value=int(flujo_base))
media_1h = (lag_1 + lag_2 + lag_4) / 3.0

# 4. Inferencia con XGBoost
input_dict = {
    "hora": hora_sel,
    "dia_semana": dia_semana,
    "feriado": feriado,
    "precipitacion": precipitacion,
    "temperatura": temperatura,
    "lag_1": lag_1,
    "lag_2": lag_2,
    "lag_4": lag_4,
    "lag_dia": lag_dia,
    "media_1h": media_1h
}

for col in cols_via:
    if f"via_{via_clean}" == col:
        input_dict[col] = 1.0
    else:
        input_dict[col] = 0.0

df_input = pd.DataFrame([input_dict])[features]

# Medir latencia de inferencia
t0 = time.perf_counter()
pred_volumen = float(modelo.predict(df_input)[0])
t1 = time.perf_counter()
latencia_ms = (t1 - t0) * 1000

pred_volumen = max(30.0, pred_volumen)
pred_vhmd = pred_volumen * 4.0

# Clasificación de Nivel de Congestión
if pred_volumen < umb_via["P50"]:
    nivel_congest = "Bajo"
    badge_html = '<span class="status-badge-green">🟢 BAJO FLUIDO</span>'
    los_estimado = "Nivel de Servicio B / C"
    semaforo_rec = "Ciclo Semafórico Base (60 s). Prioridad normal de fase verde."
    color_grafico = "#10B981"
elif pred_volumen < umb_via["P85"]:
    nivel_congest = "Moderado"
    badge_html = '<span class="status-badge-yellow">🟡 MODERADO / REGULAR</span>'
    los_estimado = "Nivel de Servicio D"
    semaforo_rec = "Ciclo Semafórico Intermedio (80 s). Extender verde en eje principal +10 s."
    color_grafico = "#F59E0B"
else:
    nivel_congest = "Alto"
    badge_html = '<span class="status-badge-red">🔴 CONGESTIÓN ALTA (CRÍTICO)</span>'
    los_estimado = "Nivel de Servicio E / F (Saturación)"
    semaforo_rec = "Ciclo Extendido de Saturación (100 s). Fase verde máxima (+25 s) y alerta de desvío."
    color_grafico = "#EF4444"

# 5. Renderizado del Dashboard Principal
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

with col_kpi1:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Demanda Prevista (15 min)", f"{pred_volumen:.0f} veh", delta=f"{pred_volumen - lag_1:+.0f} vs hace 15m")
    st.markdown('</div>', unsafe_allow_html=True)

with col_kpi2:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Tasa Horaria Equivalente", f"{pred_vhmd:.0f} veh/h", delta=f"FHP est.: {umb_via['FHP']:.3f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col_kpi3:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.write("**Nivel de Congestión:**")
    st.markdown(badge_html, unsafe_allow_html=True)
    st.caption(f"Capacidad máx: {umb_via['capacidad_max']} veh/15m")
    st.markdown('</div>', unsafe_allow_html=True)

with col_kpi4:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Latencia de Inferencia", f"{latencia_ms:.2f} ms", delta="Tiempo Real (<2s)")
    st.caption(los_estimado)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Fila 2: Recomendación de Semaforización Adaptativa
st.subheader("🚥 Recomendación para el Centro de Control de Tránsito (Cajamarca)")
info_box = f"""
* **Corredor Evaluado:** {corredor_sel}
* **Horario Consultado:** {hora_txt} hrs | **Día:** {dia_nom}
* **Directiva de Semaforización:** **{semaforo_rec}**
* **Operatividad Vial:** {los_estimado} con saturación estimada al {(pred_volumen / umb_via['capacidad_max']) * 100:.1f} % de la capacidad geométrica.
"""
st.info(info_box)

# Fila 3: Gráficos de Análisis Predictivo
col_g1, col_g2 = st.columns([6, 4])

with col_g1:
    st.subheader("📈 Proyección del Perfil Diario de Tránsito")
    
    # Generar curva horaria simulada para el día completo
    horas_dia = np.arange(6.5, 20.25, 0.25)
    flujos_sim = []
    for h in horas_dia:
        row_sim = input_dict.copy()
        row_sim["hora"] = h
        if 7.25 <= h <= 8.75 or 12.75 <= h <= 14.0 or 18.0 <= h <= 19.5:
            base_h = umb_via["P85"] * 0.95
        else:
            base_h = umb_via["P50"] * 0.80
        row_sim["lag_1"] = base_h
        row_sim["lag_2"] = base_h * 0.95
        row_sim["lag_4"] = base_h * 0.90
        row_sim["media_1h"] = base_h * 0.95
        df_sim = pd.DataFrame([row_sim])[features]
        flujos_sim.append(float(modelo.predict(df_sim)[0]))
        
    horas_etiquetas = [f"{int(h):02d}:{int(round((h-int(h))*60)):02d}" for h in horas_dia]
    
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=horas_etiquetas, y=flujos_sim,
        mode="lines", name="Demanda Proyectada",
        line=dict(color="#2563EB", width=3)
    ))
    # Líneas de umbrales
    fig_line.add_hline(y=umb_via["P85"], line_dash="dash", line_color="#DC2626", annotation_text="Congestión Alta (P85)")
    fig_line.add_hline(y=umb_via["P50"], line_dash="dot", line_color="#F59E0B", annotation_text="Mediana (P50)")
    
    # Punto actual
    fig_line.add_trace(go.Scatter(
        x=[hora_txt], y=[pred_volumen],
        mode="markers", name="Punto Actual",
        marker=dict(color=color_grafico, size=14, symbol="diamond", line=dict(color="black", width=1.5))
    ))
    
    fig_line.update_layout(
        title=f"Curva de Carga Horaria Estimada — {corredor_sel}",
        xaxis_title="Hora del Día",
        yaxis_title="Vehículos / 15 minutos",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        height=380
    )
    st.plotly_chart(fig_line, use_container_width=True)

with col_g2:
    st.subheader("🛵 Desglose Modal del Tránsito")
    
    modal_df = pd.DataFrame([
        {"Tipo": k, "Porcentaje": v, "Volumen": pred_volumen * (v / 100.0)}
        for k, v in composicion_modal.items()
    ])
    
    fig_pie = px.pie(
        modal_df,
        names="Tipo",
        values="Volumen",
        title="Composición Vehicular Estimada en Cajamarca",
        color_discrete_sequence=["#F59E0B", "#EF4444", "#3B82F6", "#10B981", "#8B5CF6"]
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

# Pestañas de soporte técnico y académico
with st.expander("📚 Ficha Técnica del Modelo y Métricas Universitarias"):
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(r"""
        **Arquitectura y Entrenamiento:**
        * **Algoritmo:** XGBoost Regressor (Extreme Gradient Boosting)
        * **Hiperparámetros:** Estimadores = 450, Learning Rate = 0.035, Max Depth = 5, L2 Reg ($\lambda$) = 2.5.
        * **Dataset Real:** 4 212 registros de aforo continuo de Cajamarca (UPN 2023, UNC 2022).
        * **Variables Exógenas:** Temperatura y Precipitación horaria de la API de Open-Meteo (-7.1638, -78.5003).
        * **Validación:** Partición cronológica estricta sin fuga temporal (70 % train, 15 % val, 15 % test).
        """)
    with col_t2:
        st.markdown(r"""
        **Métricas Oficiales en Prueba (780 intervalos):**
        * **Coeficiente $R^2$:** **0.9826** (Criterio de éxito $\ge 0.85$ ✅)
        * **Error MAPE:** **3.47 %** (Criterio de éxito $\le 15.0\%$ ✅)
        * **Error RMSE:** **21.22 veh/15 min** (Reducción de 36.86 % frente a base lineal ✅)
        * **Brecha de Generalización:** **3.54 %** (Sin sobreajuste ✅)
        * **Latencia de Inferencia:** **0.0042 s** ($< 2.0\text{ s}$ ✅)
        """)
    st.caption("Proyecto de Investigación — Grupo 6: Luis Asencio, José Muñoz, Maria Quispe, Rodrigo Sánchez, Yeber Tasilla. Docente: Carlos Mendoza. Universidad Privada del Norte (UPN).")
