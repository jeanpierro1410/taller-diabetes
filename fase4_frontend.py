# =============================================================================
# TALLER 1.1 - EQUIPO 3: DIABETES (Scikit-learn)
# FASE IV: Despliegue - FRONTEND (Streamlit) - VERSIÓN MEJORADA
# =============================================================================
# Cambio principal: el usuario ingresa valores reales (edad en años,
# peso en kg, presión en mmHg, etc.) y el código los convierte
# internamente a los valores normalizados que espera el modelo.
#
# Instrucciones:
#   1. pip install streamlit requests pandas numpy matplotlib seaborn scikit-learn joblib
#   2. Ejecutar: streamlit run fase4_frontend.py
# =============================================================================

import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

# =============================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Predictor de Diabetes | Taller 1.1",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; color: #1f4e79;
        text-align: center; font-weight: bold; padding: 1rem 0;
    }
    .subheader {
        font-size: 1.1rem; color: #555;
        text-align: center; margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%; height: 3rem; font-size: 1.1rem;
        border-radius: 8px;
        background: linear-gradient(135deg, #1f4e79, #2980b9);
        color: white; border: none;
    }
    .aviso-box {
        background: #eaf4fb; border-left: 4px solid #2980b9;
        padding: 0.8rem 1rem; border-radius: 6px;
        font-size: 0.88rem; color: #1a5276; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 2. PARÁMETROS DE NORMALIZACIÓN
# =============================================================================
# El dataset de scikit-learn tiene los valores normalizados.
# Para convertir valores reales → normalizados usamos la fórmula:
#   normalizado = (valor_real - media) / std * SCALE_FACTOR
#
# Estos parámetros fueron estimados a partir de los rangos publicados
# del dataset original de Efron et al. (2004).
# =============================================================================

NORM_PARAMS = {
    # variable: (media_real, std_real)
    'age': (48.5,  13.1),
    'bmi': (26.4,   4.4),
    'bp':  (94.6,  13.8),
    's1':  (189.1, 34.6),
    's2':  (115.4, 30.4),
    's3':  (49.8,  12.9),
    's4':  (4.07,   1.29),
    's5':  (4.64,   0.52),
    's6':  (91.3,  11.5),
}

# Factor de escala para mapear al rango [-0.2, 0.2] del dataset sklearn
SCALE_FACTOR = 0.0476

def normalizar(variable, valor_real):
    """Convierte un valor real a su equivalente normalizado para el modelo."""
    media, std = NORM_PARAMS[variable]
    return round((valor_real - media) / std * SCALE_FACTOR, 5)


# =============================================================================
# 3. FUNCIONES AUXILIARES
# =============================================================================

@st.cache_resource
def cargar_modelo_local():
    """Carga el modelo entrenado o lo entrena si no existe el archivo .pkl"""
    if os.path.exists('mejor_modelo.pkl'):
        return joblib.load('mejor_modelo.pkl')
    # Si no hay modelo guardado, entrena uno de respaldo
    diabetes = load_diabetes()
    X = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    y = diabetes.target
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return modelo

@st.cache_data
def cargar_datos():
    """Carga el dataset de diabetes de scikit-learn."""
    diabetes = load_diabetes()
    X = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target, name='target')
    return X, y, diabetes.feature_names

def categorizar_prediccion(valor):
    """Clasifica el valor predicho en un nivel de riesgo."""
    if valor < 100:
        return "Bajo", "#27ae60", "🟢"
    elif valor < 175:
        return "Moderado", "#f39c12", "🟡"
    elif valor < 250:
        return "Alto", "#e74c3c", "🔴"
    else:
        return "Muy Alto", "#8e44ad", "🟣"

def predecir_con_api(datos_norm: dict, url_api: str) -> dict:
    """Envía los datos normalizados al backend FastAPI y retorna la predicción."""
    try:
        response = requests.post(f"{url_api}/predecir", json=datos_norm, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "No se pudo conectar al backend."}
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# 4. SIDEBAR: FORMULARIO CON VALORES REALES E INTUITIVOS
# =============================================================================
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/"
        "Scikit_learn_logo_small.svg/200px-Scikit_learn_logo_small.svg.png",
        width=130
    )
    st.markdown("## ⚙️ Configuración")
    modo = st.radio(
        "Modo de predicción:",
        ["🖥️ Local (sin backend)", "🌐 API Backend"],
        index=0,
        help="Local: predice directamente en Streamlit.\nAPI: conecta al backend FastAPI."
    )
    url_api = "http://localhost:8000"
    if modo == "🌐 API Backend":
        url_api = st.text_input("URL del backend:", value="http://localhost:8000")

    st.markdown("---")
    st.markdown("## 🧑‍⚕️ Datos del Paciente")
    st.markdown(
        '<div class="aviso-box">Ingresa los valores reales del paciente. '
        'La conversión al formato del modelo se hace automáticamente.</div>',
        unsafe_allow_html=True
    )

    # ── EDAD: número entero en años ───────────────────────────────────────
    edad = st.number_input(
        "👤 Edad (años)",
        min_value=18, max_value=80, value=49, step=1,
        help="Edad del paciente en años completos."
    )

    # ── SEXO: selector intuitivo ──────────────────────────────────────────
    sexo_label = st.selectbox(
        "⚥ Sexo biológico",
        options=["Mujer", "Hombre"],
        help="Sexo biológico del paciente."
    )
    # En el dataset original: Mujer ≈ -0.044, Hombre ≈ +0.051
    sexo_norm = 0.051 if sexo_label == "Hombre" else -0.044

    # ── IMC: kg/m² con indicador de categoría ────────────────────────────
    bmi_val = st.slider(
        "⚖️ IMC - Índice de Masa Corporal (kg/m²)",
        min_value=15.0, max_value=50.0, value=26.4, step=0.1,
        help="Peso(kg) / Talla²(m). Normal: 18.5 - 24.9"
    )
    if bmi_val < 18.5:
        st.caption("🔵 Bajo peso")
    elif bmi_val < 25.0:
        st.caption("🟢 Peso normal")
    elif bmi_val < 30.0:
        st.caption("🟡 Sobrepeso")
    else:
        st.caption("🔴 Obesidad")

    # ── PRESIÓN ARTERIAL: mmHg ────────────────────────────────────────────
    bp_val = st.slider(
        "💉 Presión Arterial Media (mmHg)",
        min_value=60, max_value=140, value=95, step=1,
        help="Presión arterial media. Normal: 70 - 100 mmHg"
    )
    if bp_val < 70:
        st.caption("🔵 Hipotensión")
    elif bp_val <= 100:
        st.caption("🟢 Normal")
    elif bp_val <= 120:
        st.caption("🟡 Elevada")
    else:
        st.caption("🔴 Hipertensión")

    # ── PERFIL LIPÍDICO ───────────────────────────────────────────────────
    st.markdown("##### 🧪 Perfil Lipídico en Sangre")

    s1_val = st.slider(
        "Colesterol Total (mg/dL)",
        min_value=100, max_value=300, value=189, step=1,
        help="Deseable: < 200 mg/dL. Límite alto: 200-239. Alto: ≥ 240."
    )
    s2_val = st.slider(
        "LDL - Colesterol 'malo' (mg/dL)",
        min_value=50, max_value=250, value=115, step=1,
        help="Óptimo: < 100. Límite alto: 130-159. Alto: ≥ 160."
    )
    s3_val = st.slider(
        "HDL - Colesterol 'bueno' (mg/dL)",
        min_value=20, max_value=100, value=50, step=1,
        help="Bajo (riesgo): < 40. Bueno: ≥ 60 mg/dL."
    )
    s4_val = st.slider(
        "Relación Colesterol Total / HDL",
        min_value=1.0, max_value=10.0, value=4.1, step=0.1,
        help="Ideal: < 4.0. Mayor ratio = mayor riesgo cardiovascular."
    )
    s5_val = st.slider(
        "Log Triglicéridos (log mg/dL)",
        min_value=3.5, max_value=6.5, value=4.6, step=0.05,
        help="Logaritmo natural de triglicéridos. Normal ≈ 4.5 (≈ 90 mg/dL)."
    )

    # ── GLUCOSA ───────────────────────────────────────────────────────────
    s6_val = st.slider(
        "🩸 Glucosa en Sangre (mg/dL)",
        min_value=60, max_value=200, value=91, step=1,
        help="Ayunas normal: 70-99. Prediabetes: 100-125. Diabetes: ≥ 126."
    )
    if s6_val < 100:
        st.caption("🟢 Normal")
    elif s6_val < 126:
        st.caption("🟡 Prediabetes")
    else:
        st.caption("🔴 Rango diabético")

    # ── Convertir TODOS los valores reales a normalizados ─────────────────
    # Esta conversión ocurre silenciosamente antes de enviar al modelo
    datos_normalizados = {
        'age': normalizar('age', edad),
        'sex': sexo_norm,
        'bmi': normalizar('bmi', bmi_val),
        'bp':  normalizar('bp',  bp_val),
        's1':  normalizar('s1',  s1_val),
        's2':  normalizar('s2',  s2_val),
        's3':  normalizar('s3',  s3_val),
        's4':  normalizar('s4',  s4_val),
        's5':  normalizar('s5',  s5_val),
        's6':  normalizar('s6',  s6_val),
    }

    boton_predecir = st.button("🔍 PREDECIR PROGRESIÓN", type="primary")


# =============================================================================
# 5. CONTENIDO PRINCIPAL
# =============================================================================
st.markdown('<div class="main-header">🩺 Predictor de Progresión de Diabetes</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subheader">Taller 1.1 · Equipo 3 · Modelo de Regresión Lineal</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📍 Predicción", "📊 Análisis del Modelo", "ℹ️ Información"])


# ─────────────────────────────────────────────
# TAB 1: PREDICCIÓN
# ─────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📋 Resumen del Paciente")
        # Mostramos los valores REALES para que sea legible por cualquier persona
        resumen = pd.DataFrame({
            "Variable": [
                "👤 Edad", "⚥ Sexo", "⚖️ IMC",
                "💉 Presión Arterial", "🧪 Colesterol Total",
                "🧪 LDL", "🧪 HDL", "🧪 Col/HDL",
                "🧪 Log Triglicéridos", "🩸 Glucosa"
            ],
            "Valor ingresado": [
                f"{edad} años", sexo_label,
                f"{bmi_val} kg/m²", f"{bp_val} mmHg",
                f"{s1_val} mg/dL", f"{s2_val} mg/dL",
                f"{s3_val} mg/dL", f"{s4_val}",
                f"{s5_val}", f"{s6_val} mg/dL"
            ],
            "Valor normalizado": list(datos_normalizados.values())
        })
        st.dataframe(resumen, use_container_width=True, hide_index=True)
        st.markdown(
            '<div class="aviso-box">💡 Los <b>valores normalizados</b> son los que '
            'recibe internamente el modelo. La conversión es automática.</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("### 🎯 Resultado de la Predicción")

        if boton_predecir:
            with st.spinner("Calculando..."):

                if modo == "🖥️ Local (sin backend)":
                    modelo_local = cargar_modelo_local()
                    X_input = pd.DataFrame([datos_normalizados])
                    prediccion_val = float(modelo_local.predict(X_input)[0])
                    prediccion_val = max(25.0, min(346.0, prediccion_val))  # Limitar al rango válido
                    nivel, color_nivel, emoji_nivel = categorizar_prediccion(prediccion_val)

                    st.success("✅ Predicción completada (modo local)")

                    # Tarjeta principal del resultado
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #1f4e79, #2c7fb8);
                                padding: 2rem; border-radius: 15px; color: white; text-align: center;'>
                        <p style='font-size:1rem; opacity:0.85; margin:0;'>
                            Progresión de Diabetes Estimada</p>
                        <h1 style='font-size: 3.5rem; margin: 0.3rem 0; font-weight: bold;'>
                            {prediccion_val:.1f}
                        </h1>
                        <p style='font-size: 0.85rem; opacity: 0.75; margin:0;'>
                            escala: 25 (mínima progresión) → 346 (máxima progresión)
                        </p>
                        <hr style='opacity: 0.3; margin: 1rem 0;'>
                        <h3 style='margin:0;'>
                            {emoji_nivel} Nivel: <span style='color:#ffd700;'>{nivel}</span>
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Barra de riesgo visual
                    fig_gauge, ax = plt.subplots(figsize=(7, 1.8))
                    fig_gauge.patch.set_facecolor('#f8f9fa')
                    ax.set_facecolor('#f8f9fa')
                    ax.set_xlim(25, 346)
                    ax.set_ylim(0, 1)
                    ax.barh(0.5, 75,  left=25,  height=0.35, color='#27ae60', alpha=0.85)
                    ax.barh(0.5, 75,  left=100, height=0.35, color='#f39c12', alpha=0.85)
                    ax.barh(0.5, 75,  left=175, height=0.35, color='#e74c3c', alpha=0.85)
                    ax.barh(0.5, 96,  left=250, height=0.35, color='#8e44ad', alpha=0.85)
                    for x_lbl, txt in [(62,'Bajo'),(137,'Mod.'),(212,'Alto'),(298,'Muy\nAlto')]:
                        ax.text(x_lbl, 0.5, txt, ha='center', va='center',
                                fontsize=7, color='white', fontweight='bold')
                    ax.axvline(prediccion_val, color='black', linewidth=3, zorder=5)
                    ax.annotate(f'▼ {prediccion_val:.0f}',
                                xy=(prediccion_val, 0.88),
                                ha='center', fontsize=11, fontweight='bold')
                    ax.set_xlabel('Escala de Progresión (25 - 346)', fontsize=9)
                    ax.set_yticks([])
                    ax.set_title('Posición en la escala de riesgo', fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig_gauge)

                    # Interpretación en lenguaje natural
                    mensajes = {
                        "Bajo":     "✅ Progresión **baja**. Mantener hábitos saludables y controles anuales.",
                        "Moderado": "⚠️ Progresión **moderada**. Seguimiento médico semestral recomendado.",
                        "Alto":     "🚨 Progresión **significativa**. Se requiere atención médica activa.",
                        "Muy Alto": "🆘 Progresión **severa**. Intervención médica urgente recomendada."
                    }
                    st.info(mensajes[nivel])

                else:  # Modo API Backend
                    resultado = predecir_con_api(datos_normalizados, url_api)
                    if "error" in resultado:
                        st.error(f"❌ {resultado['error']}")
                        st.info("Asegúrate de que el backend esté corriendo:\n"
                                "`uvicorn fase4_backend:app --reload`")
                    else:
                        prediccion_val = resultado['prediccion']
                        nivel = resultado['nivel']
                        emoji_nivel = {"Bajo":"🟢","Moderado":"🟡",
                                       "Alto":"🔴","Muy Alto":"🟣"}.get(nivel, "⚪")
                        st.success("✅ Respuesta recibida del backend")
                        st.metric("Progresión Predicha", f"{prediccion_val:.1f}")
                        st.markdown(f"**Nivel:** {emoji_nivel} {nivel}")
                        st.info(resultado.get('descripcion', ''))
        else:
            st.markdown("""
            <div style='background:#f0f7ff; border-radius:10px;
                        padding:2rem; text-align:center; color:#555;'>
                <h3>👈 Completa los datos del paciente</h3>
                <p>Ajusta las variables en el panel izquierdo<br>
                y presiona <b>PREDECIR PROGRESIÓN</b></p>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 2: ANÁLISIS DEL MODELO
# ─────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Análisis Comparativo de Modelos")

    X, y, feat_names = cargar_datos()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelos_tab = {
        'Lineal Múltiple': LinearRegression(),
        'Polinomial Grado 2': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('reg', LinearRegression())
        ]),
        'Polinomial Grado 3': Pipeline([
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),
            ('reg', LinearRegression())
        ])
    }

    metricas = {}
    for nombre, mod in modelos_tab.items():
        mod.fit(X_train, y_train)
        y_pred = mod.predict(X_test)
        metricas[nombre] = {
            'R² Test': round(r2_score(y_test, y_pred), 4),
            'RMSE Test': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            'y_pred': y_pred
        }

    df_metricas = pd.DataFrame(metricas).T[['R² Test', 'RMSE Test']]
    st.dataframe(
        df_metricas.style
            .highlight_max(subset=['R² Test'], color='#d4edda')
            .highlight_min(subset=['RMSE Test'], color='#d4edda'),
        use_container_width=True
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Valor Real vs Valor Predicho (Test Set)', fontweight='bold')
    colores_plot = ['#4C72B0', '#55A868', '#C44E52']
    for i, (nombre, metr) in enumerate(metricas.items()):
        ax = axes[i]
        ax.scatter(y_test, metr['y_pred'], alpha=0.4, color=colores_plot[i], s=25)
        lim = [min(y_test.min(), metr['y_pred'].min()),
               max(y_test.max(), metr['y_pred'].max())]
        ax.plot(lim, lim, 'k--', linewidth=1.5, label='Predicción perfecta')
        ax.set_xlabel('Valor Real')
        ax.set_ylabel('Valor Predicho')
        ax.set_title(f"{nombre}\nR²={metr['R² Test']}", fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### 🔗 Correlación de Variables con el Target")
    corr = X.join(y).corr()['target'].drop('target').sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    colors_corr = ['#27ae60' if v > 0 else '#e74c3c' for v in corr.values]
    ax2.bar(corr.index, corr.values, color=colors_corr, edgecolor='white')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_title('Correlación de Variables con el Target', fontweight='bold')
    ax2.set_ylabel('Correlación de Pearson')
    plt.tight_layout()
    st.pyplot(fig2)


# ─────────────────────────────────────────────
# TAB 3: INFORMACIÓN
# ─────────────────────────────────────────────
with tab3:
    st.markdown("""
    ## ℹ️ Sobre este Proyecto

    **Taller 1.1 · Equipo 3 · Dataset: Diabetes (Scikit-learn)**

    ### 🎯 Objetivo
    Predecir la progresión de la diabetes un año después del diagnóstico,
    usando 10 variables clínicas del paciente.

    ### 📦 Dataset
    - **Fuente:** `sklearn.datasets.load_diabetes` (Efron et al., 2004)
    - **Muestras:** 442 pacientes
    - **Variables:** 10 (el modelo las recibe normalizadas internamente)
    - **Target:** Progresión cuantitativa (escala 25 - 346)

    ### 🔄 Metodología CRISP-DM
    | Fase | Archivo | Descripción |
    |------|---------|-------------|
    | **I. EDA** | `fase1_eda.py` | Análisis exploratorio, matrices de correlación |
    | **II. Modelado** | `fase2_modelado.py` | Regresión Lineal, Poly-2, Poly-3 |
    | **III. Evaluación** | `fase3_evaluacion.py` | R², RMSE, residuos, curvas de aprendizaje |
    | **IV. Despliegue** | `fase4_backend.py` + este archivo | FastAPI + Streamlit |

    ### 📐 Variables y sus Unidades Reales
    | Variable | Descripción | Unidad | Rango típico |
    |----------|-------------|--------|--------------|
    | age | Edad | años | 18 - 80 |
    | sex | Sexo biológico | Hombre/Mujer | — |
    | bmi | Índice de Masa Corporal | kg/m² | 15 - 50 |
    | bp | Presión arterial media | mmHg | 60 - 140 |
    | s1 | Colesterol total | mg/dL | 100 - 300 |
    | s2 | LDL (colesterol malo) | mg/dL | 50 - 250 |
    | s3 | HDL (colesterol bueno) | mg/dL | 20 - 100 |
    | s4 | Relación Colesterol/HDL | ratio | 1 - 10 |
    | s5 | Log de triglicéridos | log(mg/dL) | 3.5 - 6.5 |
    | s6 | Glucosa en sangre | mg/dL | 60 - 200 |

    ### 🚀 Hosting gratuito
    - **Backend (FastAPI):** [Render.com](https://render.com) o [Railway.app](https://railway.app)
    - **Frontend (Streamlit):** [Streamlit Cloud](https://share.streamlit.io)
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "Taller 1.1 · Equipo 3 · Modelos de Regresión Lineal · 2025 &nbsp;|&nbsp; "
    "Dataset: Efron et al. (2004)"
    "</div>",
    unsafe_allow_html=True
)