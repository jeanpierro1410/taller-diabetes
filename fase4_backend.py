# =============================================================================
# TALLER 1.1 - EQUIPO 3: DIABETES (Scikit-learn)
# FASE IV: Despliegue - BACKEND (FastAPI)
# =============================================================================
# Archivo: fase4_backend.py
#
# Instrucciones para ejecutar:
#   1. pip install fastapi uvicorn scikit-learn joblib numpy pandas
#   2. Primero ejecuta fase2_modelado.py para generar "mejor_modelo.pkl"
#   3. Luego ejecuta: uvicorn fase4_backend:app --reload --port 8000
#   4. Prueba en: http://localhost:8000/docs  (Swagger UI automático)
# =============================================================================

# --- IMPORTACIONES ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Permite peticiones desde el frontend
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd
import joblib
import os

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


# =============================================================================
# 1. CONFIGURACIÓN DE LA APLICACIÓN FASTAPI
# =============================================================================
app = FastAPI(
    title="API de Predicción de Diabetes",
    description="""
    ## Taller 1.1 - Equipo 3

    Esta API predice la **progresión de la diabetes** un año después del diagnóstico,
    basándose en 10 variables clínicas del paciente.

    ### Variables de entrada:
    - **age**: Edad (normalizada)
    - **sex**: Sexo (normalizado)
    - **bmi**: Índice de Masa Corporal (normalizado)
    - **bp**: Presión arterial media (normalizada)
    - **s1 - s6**: Valores séricos de lípidos y glucosa (normalizados)
    """,
    version="1.0.0"
)

# Permitir peticiones desde cualquier origen (necesario para el frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # En producción, especificar el dominio exacto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 2. CARGA DEL MODELO AL INICIAR
# =============================================================================
# Intentamos cargar el modelo pre-entrenado; si no existe, lo entrenamos aquí
def cargar_o_entrenar_modelo():
    """Carga el modelo desde disco o lo entrena si no existe."""
    if os.path.exists('mejor_modelo.pkl'):
        print("📦 Cargando modelo desde 'mejor_modelo.pkl'...")
        return joblib.load('mejor_modelo.pkl')
    else:
        print("⚙️  Modelo no encontrado. Entrenando modelo lineal de respaldo...")
        diabetes = load_diabetes()
        X = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
        y = diabetes.target
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        joblib.dump(modelo, 'mejor_modelo.pkl')
        print("✅ Modelo entrenado y guardado.")
        return modelo

modelo = cargar_o_entrenar_modelo()
FEATURE_NAMES = load_diabetes().feature_names  # ['age', 'sex', 'bmi', ...]


# =============================================================================
# 3. ESQUEMA DE DATOS DE ENTRADA (Validación automática con Pydantic)
# =============================================================================
class DatosInput(BaseModel):
    """
    Esquema de validación para los datos de entrada.
    El usuario envía un JSON con estas 10 variables (ya normalizadas).
    Pydantic valida automáticamente tipos y rangos.
    """
    age: float = Field(..., ge=-0.2, le=0.2, description="Edad normalizada", example=0.038)
    sex: float = Field(..., ge=-0.2, le=0.2, description="Sexo normalizado", example=0.050)
    bmi: float = Field(..., ge=-0.2, le=0.2, description="IMC normalizado", example=0.062)
    bp:  float = Field(..., ge=-0.2, le=0.2, description="Presión arterial normalizada", example=-0.025)
    s1:  float = Field(..., ge=-0.2, le=0.2, description="Colesterol total normalizado", example=-0.044)
    s2:  float = Field(..., ge=-0.2, le=0.2, description="LDL normalizado", example=-0.034)
    s3:  float = Field(..., ge=-0.2, le=0.2, description="HDL normalizado", example=-0.043)
    s4:  float = Field(..., ge=-0.2, le=0.2, description="Colesterol/HDL normalizado", example=-0.002)
    s5:  float = Field(..., ge=-0.2, le=0.2, description="Log triglicéridos normalizado", example=0.019)
    s6:  float = Field(..., ge=-0.2, le=0.2, description="Glucosa normalizada", example=-0.017)

    class Config:
        # Ejemplo que aparece en /docs (Swagger)
        json_schema_extra = {
            "example": {
                "age": 0.038, "sex": 0.050, "bmi": 0.062, "bp": -0.025,
                "s1": -0.044, "s2": -0.034, "s3": -0.043, "s4": -0.002,
                "s5": 0.019, "s6": -0.017
            }
        }


class PrediccionOutput(BaseModel):
    """Esquema de la respuesta que devuelve la API."""
    prediccion: float          # Valor predicho (progresión de diabetes)
    nivel: str                 # Categoría: Bajo / Moderado / Alto / Muy Alto
    descripcion: str           # Interpretación en lenguaje natural
    datos_recibidos: dict      # Echo de los datos enviados (para verificación)


# =============================================================================
# 4. ENDPOINTS DE LA API
# =============================================================================

@app.get("/", tags=["Estado"])
def raiz():
    """Endpoint de verificación: confirma que la API está funcionando."""
    return {
        "mensaje": "✅ API de Predicción de Diabetes - Funcionando",
        "version": "1.0.0",
        "endpoints": {
            "POST /predecir": "Realizar una predicción",
            "GET /info-modelo": "Información del modelo",
            "GET /ejemplo": "Obtener datos de ejemplo",
            "GET /docs": "Documentación Swagger (interfaz interactiva)"
        }
    }


@app.get("/salud", tags=["Estado"])
def verificar_salud():
    """Health check - útil para servidores como Render/Railway."""
    return {"estado": "saludable", "modelo_cargado": modelo is not None}


@app.post("/predecir", response_model=PrediccionOutput, tags=["Predicción"])
def predecir(datos: DatosInput):
    """
    ## Realizar una predicción

    Recibe 10 variables clínicas normalizadas y retorna:
    - La predicción numérica (progresión de diabetes al año)
    - El nivel de riesgo (Bajo / Moderado / Alto / Muy Alto)
    - Una descripción interpretable

    **Nota:** Todas las variables deben estar normalizadas (rango aprox. -0.2 a 0.2)
    tal como las entrega el dataset de scikit-learn.
    """
    try:
        # Convertir los datos del JSON a un array numpy
        valores = [
            datos.age, datos.sex, datos.bmi, datos.bp,
            datos.s1, datos.s2, datos.s3, datos.s4, datos.s5, datos.s6
        ]
        X_input = np.array(valores).reshape(1, -1)  # Shape: (1, 10)
        X_df = pd.DataFrame(X_input, columns=FEATURE_NAMES)

        # Realizar la predicción
        prediccion = float(modelo.predict(X_df)[0])
        prediccion = round(prediccion, 2)

        # Categorizar el nivel de progresión
        if prediccion < 100:
            nivel = "Bajo"
            descripcion = "Progresión lenta de la diabetes esperada en el próximo año."
        elif prediccion < 175:
            nivel = "Moderado"
            descripcion = "Progresión moderada. Se recomienda seguimiento médico regular."
        elif prediccion < 250:
            nivel = "Alto"
            descripcion = "Progresión significativa. Se requiere atención médica activa."
        else:
            nivel = "Muy Alto"
            descripcion = "Progresión severa esperada. Intervención médica urgente recomendada."

        return PrediccionOutput(
            prediccion=prediccion,
            nivel=nivel,
            descripcion=descripcion,
            datos_recibidos=datos.dict()
        )

    except Exception as e:
        # Si algo falla, retornamos un error HTTP 500 con descripción
        raise HTTPException(status_code=500, detail=f"Error al predecir: {str(e)}")


@app.get("/info-modelo", tags=["Información"])
def info_modelo():
    """Retorna información sobre el modelo desplegado."""
    tipo = type(modelo).__name__
    if hasattr(modelo, 'steps'):  # Es un Pipeline
        tipo = f"Pipeline: {' → '.join([type(s[1]).__name__ for s in modelo.steps])}"

    return {
        "tipo_modelo": tipo,
        "variables_entrada": list(FEATURE_NAMES),
        "variable_objetivo": "Progresión de diabetes (25 - 346 aprox.)",
        "dataset": "sklearn.datasets.load_diabetes (442 muestras, 10 features)"
    }


@app.get("/ejemplo", tags=["Información"])
def obtener_ejemplo():
    """Retorna un ejemplo de datos reales del dataset para probar la API."""
    diabetes = load_diabetes()
    # Tomamos el primer paciente del dataset
    paciente_ejemplo = diabetes.data[0]
    valor_real = float(diabetes.target[0])

    datos_ejemplo = {feat: float(round(val, 4))
                     for feat, val in zip(FEATURE_NAMES, paciente_ejemplo)}

    return {
        "descripcion": "Ejemplo del primer paciente del dataset",
        "datos_para_enviar": datos_ejemplo,
        "valor_real": valor_real,
        "nota": "Usa estos datos en POST /predecir para ver la predicción"
    }


# =============================================================================
# 5. PUNTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Iniciando servidor FastAPI...")
    print("   URL local: http://localhost:8000")
    print("   Swagger UI: http://localhost:8000/docs")
    print("   ReDoc: http://localhost:8000/redoc")
    uvicorn.run("fase4_backend:app", host="0.0.0.0", port=8000, reload=True)
