import os
import time
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)

# 1. Configuración de página
st.set_page_config(
    page_title="Predicción de Tráfico — Cajamarca 2026",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .status-badge-green {
        background-color: #DEF7EC; color: #03543F;
        padding: 6px 14px; border-radius: 20px; font-weight: bold; display: inline-block;
    }
    .status-badge-yellow {
        background-color: #FEF08A; color: #854D0E;
        padding: 6px 14px; border-radius: 20px; font-weight: bold; display: inline-block;
    }
    .status-badge-red {
        background-color: #FDE8E8; color: #9B1C1C;
        padding: 6px 14px; border-radius: 20px; font-weight: bold; display: inline-block;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle():
    return joblib.load("modelo_xgboost_cajamarca.pkl")


@st.cache_data
def load_dataset():
    if os.path.exists("aforos_cajamarca.csv"):
        df = pd.read_csv("aforos_cajamarca.csv")
        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
        df["fecha"] = df["fecha_hora"].dt.date
        return df.sort_values(["via", "fecha_hora"]).reset_index(drop=True)
    return None


bundle = load_bundle()
df_data = load_dataset()

modelo = bundle["modelo"]
features = bundle["features"]
umbrales = bundle["umbrales"]
margen_alerta = float(bundle.get("margen_alerta", 0.01))
metricas_rep = bundle.get("metricas_prueba_720", {})

DIAS_DICT = {
    "Lunes": 0,
    "Martes": 1,
    "Miércoles": 2,
    "Jueves": 3,
    "Viernes": 4,
    "Sábado": 5,
    "Domingo": 6,
}

CORREDORES_LABELS = {
    "Av. Hoyos Rubio (Evitamiento Norte)": "Av. Hoyos Rubio",
    "Av. Atahualpa (Evitamiento Sur)": "Av. Atahualpa",
    "Av. Los Héroes": "Av. Los Héroes",
}

COMPOSICION_MODAL = {
    "Mototaxis": 28.0,
    "Motos lineales": 18.0,
    "Autos / Camionetas": 35.0,
    "Combis / Transporte": 11.0,
    "Pesados / Buses": 6.0,
}

FHP_REF = {
    "Av. Atahualpa": 0.946,
    "Av. Hoyos Rubio": 0.943,
    "Av. Los Héroes": 0.952,
}


def build_feature_row(corredor: str, dia_idx: int, hora_dec: float) -> pd.DataFrame:
    row = {c: 0.0 for c in features}
    row["hora"] = float(hora_dec)
    dia_col = f"dia_semana_{dia_idx}"
    via_col = f"via_{corredor}"
    if dia_col in row:
        row[dia_col] = 1.0
    if via_col in row:
        row[via_col] = 1.0
    return pd.DataFrame([row])[features]


def clasificar_nivel(vol: float, corredor: str):
    p50 = umbrales[corredor]["P50"]
    p85 = umbrales[corredor]["P85"]
    umbral_alerta = p85 * (1.0 - margen_alerta)
    if vol < p50:
        return (
            "Bajo",
            '<span class="status-badge-green">🟢 BAJO</span>',
            "Ciclo semafórico base (60 s). Operación fluida sin intervención adicional.",
            "#10B981",
        )
    elif vol < umbral_alerta:
        return (
            "Moderado",
            '<span class="status-badge-yellow">🟡 MODERADO</span>',
            "Ciclo semafórico intermedio (80 s). Extender fase verde en eje principal +10 s.",
            "#F59E0B",
        )
    else:
        return (
            "Alto",
            '<span class="status-badge-red">🔴 ALTO / ALERTA DE CONGESTIÓN</span>',
            "Ciclo extendido de saturación (100 s). Priorizar fase verde (+25 s) y monitoreo operativo.",
            "#EF4444",
        )


# Encabezado
st.markdown(
    '<div class="main-header">🚦 Sistema Predictivo de Flujo Vehicular — Cajamarca 2026</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Sistema de apoyo a la decisión basado en Machine Learning (XGBoost) para alertas tempranas de congestión</div>',
    unsafe_allow_html=True,
)

# Barra lateral
st.sidebar.header("⚙️ Parámetros de Consulta")
corredor_lbl = st.sidebar.selectbox(
    "Corredor Vial Estructurante", list(CORREDORES_LABELS.keys()), index=0
)
via_sel = CORREDORES_LABELS[corredor_lbl]
dia_nom = st.sidebar.selectbox("Día de la Semana", list(DIAS_DICT.keys()), index=3)
dia_idx = DIAS_DICT[dia_nom]

hora_sel = st.sidebar.slider(
    "Hora del Día (07:00 a 19:45)",
    min_value=7.0,
    max_value=19.75,
    value=7.5,
    step=0.25,
    format="%.2f h",
)
h_int = int(hora_sel)
m_int = int(round((hora_sel - h_int) * 60))
hora_txt = f"{h_int:02d}:{m_int:02d}"
st.sidebar.caption(f"⏰ Intervalo seleccionado: **{hora_txt}**")

st.sidebar.markdown("---")
st.sidebar.info(
    "Estimación basada en patrones horarios por corredor y día de la semana (XGBoost), "
    "sin requerir sensores en tiempo real."
)

# Pestañas principales
tab_est, tab_val = st.tabs(
    ["🔮 Estimación y Alerta Operativa", "📊 Validación con Días de Prueba (07/06–11/06/2023)"]
)

# ==============================================================================
# PESTAÑA 1: ESTIMACIÓN EN TIEMPO REAL
# ==============================================================================
with tab_est:
    X_in = build_feature_row(via_sel, dia_idx, hora_sel)
    t0 = time.perf_counter()
    pred_vol = float(modelo.predict(X_in)[0])
    lat_ms = (time.perf_counter() - t0) * 1000

    p50_v = umbrales[via_sel]["P50"]
    p85_v = umbrales[via_sel]["P85"]
    alerta_v = p85_v * (1.0 - margen_alerta)
    nivel_txt, badge_html, rec_sem, color_pt = clasificar_nivel(pred_vol, via_sel)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Volumen Estimado (15 min)", f"{pred_vol:.0f} veh")
    with c2:
        st.metric("Tasa Horaria Equivalente", f"{pred_vol * 4:.0f} veh/h", delta=f"FHP medio: {FHP_REF[via_sel]:.3f}")
    with c3:
        st.write("**Nivel de Congestión (Alerta Calibrada):**")
        st.markdown(badge_html, unsafe_allow_html=True)
        st.caption(f"P50: {p50_v:.0f} | Alerta (-{margen_alerta*100:.1f}%): {alerta_v:.0f} | P85: {p85_v:.0f}")
    with c4:
        st.metric("Latencia de Inferencia", f"{lat_ms:.2f} ms", delta="Umbral < 2 000 ms")

    st.info(
        f"**Corredor:** {corredor_lbl} | **Día:** {dia_nom} | **Hora:** {hora_txt}  \n"
        f"**Recomendación Semafórica:** {rec_sem}"
    )

    col_g1, col_g2 = st.columns([6, 4])
    with col_g1:
        horas_dia = np.arange(7.0, 20.0, 0.25)
        X_dia = pd.concat([build_feature_row(via_sel, dia_idx, h) for h in horas_dia], ignore_index=True)
        preds_dia = modelo.predict(X_dia)
        etq_horas = [f"{int(h):02d}:{int(round((h - int(h)) * 60)):02d}" for h in horas_dia]

        fig_line = go.Figure()
        fig_line.add_trace(
            go.Scatter(
                x=etq_horas, y=preds_dia, mode="lines",
                name="Perfil estimado XGBoost", line=dict(color="#2563EB", width=3)
            )
        )
        fig_line.add_hline(y=p85_v, line_dash="dash", line_color="#DC2626", annotation_text=f"P85 ({p85_v:.0f})")
        fig_line.add_hline(y=alerta_v, line_dash="dot", line_color="#EA580C", annotation_text=f"Alerta ({alerta_v:.0f})")
        fig_line.add_hline(y=p50_v, line_dash="dot", line_color="#10B981", annotation_text=f"P50 ({p50_v:.0f})")
        fig_line.add_trace(
            go.Scatter(
                x=[hora_txt], y=[pred_vol], mode="markers", name="Hora consultada",
                marker=dict(color=color_pt, size=14, symbol="diamond", line=dict(color="black", width=1.5)),
            )
        )
        fig_line.update_layout(
            title=f"Perfil Diario Estimado — {corredor_lbl} ({dia_nom})",
            xaxis_title="Hora del día",
            yaxis_title="Vehículos / 15 min",
            hovermode="x unified",
            height=380,
            margin=dict(l=20, r=20, t=45, b=20),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with col_g2:
        modal_df = pd.DataFrame(
            [{"Tipo": k, "Volumen": pred_vol * (v / 100.0)} for k, v in COMPOSICION_MODAL.items()]
        )
        fig_pie = px.pie(
            modal_df, names="Tipo", values="Volumen",
            title="Composición Modal de Referencia (Cajamarca)",
            color_discrete_sequence=["#F59E0B", "#EF4444", "#3B82F6", "#10B981", "#8B5CF6"],
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(height=380, margin=dict(l=20, r=20, t=45, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

# ==============================================================================
# PESTAÑA 2: VALIDACIÓN CON LOS 5 DÍAS DE PRUEBA (2023-06-07 a 2023-06-11)
# ==============================================================================
with tab_val:
    if df_data is None:
        st.warning("No se encontró `aforos_cajamarca.csv` para mostrar la pestaña de validación.")
    else:
        fechas_unicas = sorted(df_data["fecha"].unique())
        dias_test = fechas_unicas[22:]  # Últimos 5 días de prueba (07/06 al 11/06/2023)
        df_te = df_data[df_data["fecha"].isin(dias_test)].copy()

        # Predecir sobre todos los intervalos de prueba
        X_te = pd.concat(
            [build_feature_row(r["via"], int(r["dia_semana"]), float(r["hora"])) for _, r in df_te.iterrows()],
            ignore_index=True,
        )
        df_te["pred"] = modelo.predict(X_te)
        df_te_720 = df_te[df_te["hora"] >= 8.0]

        r2_t = r2_score(df_te_720["volumen"], df_te_720["pred"])
        mape_t = mean_absolute_percentage_error(df_te_720["volumen"], df_te_720["pred"]) * 100
        mae_t = mean_absolute_error(df_te_720["volumen"], df_te_720["pred"])
        rmse_t = root_mean_squared_error(df_te_720["volumen"], df_te_720["pred"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R² en Prueba (n = 720)", f"{r2_t:.4f}")
        m2.metric("MAPE en Prueba", f"{mape_t:.2f} %")
        m3.metric("MAE en Prueba", f"{mae_t:.2f} veh")
        m4.metric("RMSE en Prueba", f"{rmse_t:.2f} veh")

        fecha_sel = st.selectbox(
            "Seleccionar día del conjunto de prueba (no visto en entrenamiento):",
            dias_test,
            index=1 if len(dias_test) > 1 else 0,
            format_func=lambda d: d.strftime("%d/%m/%Y"),
        )
        sub_d = df_te[df_te["fecha"] == fecha_sel].sort_values(["via", "hora"])

        fig_val = go.Figure()
        colores = {"Av. Hoyos Rubio": "#D62728", "Av. Atahualpa": "#1F77B4", "Av. Los Héroes": "#2CA02C"}
        for via, g in sub_d.groupby("via"):
            etq_h = [f"{int(h):02d}:{int(round((h - int(h)) * 60)):02d}" for h in g["hora"]]
            fig_val.add_trace(
                go.Scatter(x=etq_h, y=g["volumen"], mode="lines", name=f"{via} (Observado)",
                           line=dict(color=colores[via], width=2.4))
            )
            fig_val.add_trace(
                go.Scatter(x=etq_h, y=g["pred"], mode="lines", name=f"{via} (Estimado)",
                           line=dict(color=colores[via], width=2.0, dash="dash"))
            )
        fig_val.update_layout(
            title=f"Flujo Observado vs. Estimado — Día de Prueba {fecha_sel.strftime('%d/%m/%Y')}",
            xaxis_title="Hora del día",
            yaxis_title="Volumen (veh / 15 min)",
            hovermode="x unified",
            height=420,
            margin=dict(l=20, r=20, t=45, b=20),
        )
        st.plotly_chart(fig_val, use_container_width=True)

        # Tabla desagregada por corredor
        rows_via = []
        for via in ["Av. Atahualpa", "Av. Hoyos Rubio", "Av. Los Héroes"]:
            gv = df_te_720[df_te_720["via"] == via]
            p85_c = umbrales[via]["P85"]
            al_c = p85_c * (1.0 - margen_alerta)
            ya = (gv["volumen"] >= p85_c).astype(int)
            pa = (gv["pred"] >= al_c).astype(int)
            rec_al = ((ya == 1) & (pa == 1)).sum() / max(1, ya.sum()) * 100
            rows_via.append({
                "Corredor": via,
                "R²": round(r2_score(gv["volumen"], gv["pred"]), 3),
                "MAPE %": round(mean_absolute_percentage_error(gv["volumen"], gv["pred"]) * 100, 2),
                "MAE": round(mean_absolute_error(gv["volumen"], gv["pred"]), 2),
                "RMSE": round(root_mean_squared_error(gv["volumen"], gv["pred"]), 2),
                "Int. altos (≥ P85)": int(ya.sum()),
                "Recall alerta %": round(rec_al, 1),
            })
        st.dataframe(pd.DataFrame(rows_via), use_container_width=True, hide_index=True)

# Ficha Técnica
with st.expander("📚 Ficha Técnica del Modelo Final y Transparencia Metodológica"):
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(r"""
        **Arquitectura y Datos:**
        * **Modelo final:** XGBoost Regressor (`learning_rate=0.05`, `max_depth=7`, `subsample=0.8`, `reg_lambda=1`, parada temprana 50 rondas).
        * **Variables seleccionadas (Tabla 22):** Solo calendario (**11 variables**: `hora`, `dia_semana` one-hot y `via` one-hot).
        * **Dataset:** `aforos_cajamarca.csv` (4 212 intervalos de 15 min, 27 días: 16/05 al 11/06/2023), reconstruido a partir de perfiles de aforo de referencia (García Díaz, 2023; Cusquisibán Del Campo, 2023) y clima de Open-Meteo.
        * **Partición cronológica sin fuga:** 18 días entrenamiento (2 808), 4 validación (624), 5 prueba (780; evaluación común 08:00–19:45 = 720).
        """)
    with col_t2:
        st.markdown(r"""
        **Desempeño Oficial y Alcance:**
        * **R² global en prueba (720):** **0.988** (umbral $\ge 0.85$ ✅).
        * **MAPE global:** **< 3.0 %** (umbral $\le 15.0\ \%$ ✅).
        * **Mejora sobre línea base lineal:** **> 44 %** (umbral $\ge 20.0\ \%$ ✅).
        * **Brecha de generalización:** **< 2.0 %** (umbral $\le 10.0\ \%$ ✅).
        * **Limitación declarada:** Las clases Bajo, Moderado y Alto se basan en percentiles del volumen (P50/P85) sobre un dataset semisintético y no equivalen a niveles de servicio HCM medidos en campo.
        """)
    st.caption(
        "Proyecto de Investigación — Grupo 6: Luis Asencio, José Muñoz, María Quispe, Rodrigo Sánchez, Yeber Tasilla. "
        "Docente: Carlos Mendoza. Universidad Privada del Norte (UPN)."
    )
