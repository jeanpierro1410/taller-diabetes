# =============================================================================
# TALLER 1.1 - EQUIPO 3: DIABETES (Scikit-learn)
# FASE II: Modelado (Modeling)
# =============================================================================
# Objetivo: Entrenar y comparar 3 modelos:
#   1. Regresión Lineal Múltiple (baseline)
#   2. Regresión Polinomial grado 2
#   3. Regresión Polinomial grado 3
# Y generar las visualizaciones de "Valor Real vs Valor Predicho".
# =============================================================================

# --- 1. IMPORTACIÓN DE LIBRERÍAS ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Para guardar/cargar modelos entrenados
import os

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline  # Para encadenar pasos (poly + regresión)
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuración visual
plt.rcParams['figure.facecolor'] = '#f8f9fa'
sns.set_palette("husl")

print("=" * 60)
print("  TALLER 1.1 | EQUIPO 3 | FASE II: MODELADO")
print("=" * 60)


# --- 2. CARGA Y PREPARACIÓN DE DATOS ---
diabetes = load_diabetes()
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# Separar features (X) y target (y)
X = df.drop('target', axis=1)   # Variables independientes (10 features)
y = df['target']                 # Variable a predecir

# División entrenamiento / prueba (80% - 20%)
# random_state=42 garantiza que siempre obtenemos la misma división
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,       # 20% para prueba = ~88 muestras
    random_state=42
)

print(f"\n📦 División del dataset:")
print(f"   Entrenamiento: {X_train.shape[0]} muestras ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"   Prueba:        {X_test.shape[0]} muestras  ({X_test.shape[0]/len(X)*100:.0f}%)")
print(f"   Features:      {X_train.shape[1]} variables")


# --- 3. FUNCIÓN AUXILIAR PARA EVALUAR MODELOS ---
def evaluar_modelo(nombre, modelo, X_tr, y_tr, X_te, y_te):
    """
    Entrena un modelo y retorna sus métricas de evaluación.
    
    Parámetros:
        nombre   : Nombre del modelo (string)
        modelo   : Objeto modelo de scikit-learn
        X_tr/y_tr: Datos de entrenamiento
        X_te/y_te: Datos de prueba
    
    Retorna:
        dict con métricas y predicciones
    """
    modelo.fit(X_tr, y_tr)          # Entrenamiento
    y_pred_train = modelo.predict(X_tr)   # Predicciones en entrenamiento
    y_pred_test  = modelo.predict(X_te)   # Predicciones en prueba

    # Métricas en entrenamiento
    r2_train   = r2_score(y_tr, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_tr, y_pred_train))

    # Métricas en prueba (las que más importan para evaluar generalización)
    r2_test   = r2_score(y_te, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_te, y_pred_test))

    # Validación cruzada (5-fold) para una evaluación más robusta
    cv_scores = cross_val_score(modelo, X_tr, y_tr, cv=5, scoring='r2')

    print(f"\n🔹 {nombre}")
    print(f"   R²  (Train): {r2_train:.4f}  |  R²  (Test): {r2_test:.4f}")
    print(f"   RMSE(Train): {rmse_train:.2f}  |  RMSE(Test): {rmse_test:.2f}")
    print(f"   CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Detectar overfitting: si R² train >> R² test, hay overfitting
    gap = r2_train - r2_test
    if gap > 0.10:
        print(f"   ⚠️  POSIBLE OVERFITTING (gap R²: {gap:.3f})")
    else:
        print(f"   ✅ Buen ajuste (gap R²: {gap:.3f})")

    return {
        'nombre': nombre,
        'modelo': modelo,
        'y_pred_test': y_pred_test,
        'y_pred_train': y_pred_train,
        'r2_train': r2_train, 'r2_test': r2_test,
        'rmse_train': rmse_train, 'rmse_test': rmse_test,
        'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()
    }


# --- 4. MODELO 1: REGRESIÓN LINEAL MÚLTIPLE (BASELINE) ---
print("\n" + "─" * 60)
print("  MODELO 1: Regresión Lineal Múltiple (Baseline)")
print("─" * 60)

modelo_lineal = LinearRegression()
res_lineal = evaluar_modelo(
    "Regresión Lineal Múltiple",
    modelo_lineal, X_train, y_train, X_test, y_test
)

# Mostrar coeficientes del modelo lineal
print("\n   📐 Coeficientes del modelo lineal:")
coef_df = pd.DataFrame({
    'Variable': X.columns,
    'Coeficiente': modelo_lineal.coef_
}).sort_values('Coeficiente', key=abs, ascending=False)
for _, row in coef_df.iterrows():
    signo = "↑" if row['Coeficiente'] > 0 else "↓"
    print(f"      {signo} {row['Variable']:6s}: {row['Coeficiente']:+.2f}")
print(f"   Intercepto: {modelo_lineal.intercept_:.2f}")


# --- 5. MODELO 2: REGRESIÓN POLINOMIAL GRADO 2 ---
print("\n" + "─" * 60)
print("  MODELO 2: Regresión Polinomial Grado 2")
print("─" * 60)
print("  💡 PolynomialFeatures genera nuevas features como x²,")
print("     x₁·x₂, etc. (términos cuadráticos e interacciones)")

# Usamos Pipeline para automatizar: transformación polinomial → regresión
modelo_poly2 = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    # include_bias=False porque LinearRegression ya incluye intercepto
    ('regresion', LinearRegression())
])
res_poly2 = evaluar_modelo(
    "Regresión Polinomial Grado 2",
    modelo_poly2, X_train, y_train, X_test, y_test
)
# ¿Cuántas features genera grado 2?
pf2 = PolynomialFeatures(degree=2, include_bias=False)
pf2.fit(X_train)
print(f"\n   📌 Features originales: {X_train.shape[1]}")
print(f"   📌 Features tras grado 2: {pf2.n_output_features_} (incluye x², x₁x₂, etc.)")


# --- 6. MODELO 3: REGRESIÓN POLINOMIAL GRADO 3 ---
print("\n" + "─" * 60)
print("  MODELO 3: Regresión Polinomial Grado 3")
print("─" * 60)
print("  ⚠️  Con grado 3 aumenta mucho la complejidad.")
print("     Es importante vigilar el overfitting.")

modelo_poly3 = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('regresion', LinearRegression())
])
res_poly3 = evaluar_modelo(
    "Regresión Polinomial Grado 3",
    modelo_poly3, X_train, y_train, X_test, y_test
)
pf3 = PolynomialFeatures(degree=3, include_bias=False)
pf3.fit(X_train)
print(f"\n   📌 Features tras grado 3: {pf3.n_output_features_}")


# --- 7. COMPARACIÓN GLOBAL DE MODELOS ---
print("\n" + "=" * 60)
print("  COMPARATIVA FINAL DE MODELOS")
print("=" * 60)
resultados = [res_lineal, res_poly2, res_poly3]
tabla = pd.DataFrame([{
    'Modelo': r['nombre'],
    'R² Test': round(r['r2_test'], 4),
    'RMSE Test': round(r['rmse_test'], 2),
    'R² CV (mean)': round(r['cv_mean'], 4),
    'Gap Overfit': round(r['r2_train'] - r['r2_test'], 4)
} for r in resultados])
print(tabla.to_string(index=False))
mejor = tabla.loc[tabla['R² Test'].idxmax(), 'Modelo']
print(f"\n🏆 Mejor modelo (mayor R² en test): {mejor}")


# --- 8. VISUALIZACIÓN: REAL vs PREDICHO ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Valor Real vs Valor Predicho (Test Set)',
             fontsize=14, fontweight='bold')
colores = ['#4C72B0', '#55A868', '#C44E52']
titulos = ['Lineal Múltiple', 'Polinomial Grado 2', 'Polinomial Grado 3']

for i, (res, color, titulo) in enumerate(zip(resultados, colores, titulos)):
    y_pred = res['y_pred_test']

    # Scatter: puntos reales vs predichos
    axes[i].scatter(y_test, y_pred, alpha=0.5, color=color, s=30, label='Predicciones')

    # Línea perfecta (y_real = y_pred): si los puntos están sobre esta línea, el modelo es perfecto
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'k--',
                 linewidth=2, label='Predicción perfecta')

    axes[i].set_xlabel('Valor Real', fontsize=11)
    axes[i].set_ylabel('Valor Predicho', fontsize=11)
    axes[i].set_title(
        f'{titulo}\nR²={res["r2_test"]:.4f} | RMSE={res["rmse_test"]:.2f}',
        fontweight='bold'
    )
    axes[i].legend(fontsize=9)
    axes[i].set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('modelado_01_real_vs_predicho.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Gráfica guardada: modelado_01_real_vs_predicho.png")


# --- 9. VISUALIZACIÓN: CURVA DE AJUSTE POR VARIABLE ---
# Mostramos cómo cada modelo se ajusta a bmi (la variable más correlacionada)
variable_principal = 'bmi'  # Más alta correlación con el target

# Creamos un rango de valores para graficar la curva
X_bmi_range = np.linspace(X[variable_principal].min(),
                           X[variable_principal].max(), 200).reshape(-1, 1)

# Necesitamos 10 features para los modelos, así que fijamos las demás en su media
X_media = X.mean().values  # Media de todas las features
X_plot = np.tile(X_media, (200, 1))  # Repetimos 200 veces
idx_bmi = list(X.columns).index(variable_principal)
X_plot[:, idx_bmi] = X_bmi_range.flatten()  # Solo variamos bmi
X_plot_df = pd.DataFrame(X_plot, columns=X.columns)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Curva de Ajuste: Variable "{variable_principal}" vs Target\n'
             f'(demás variables fijadas en su media)',
             fontsize=13, fontweight='bold')

for i, (res, color, titulo) in enumerate(zip(resultados, colores, titulos)):
    # Puntos reales
    axes[i].scatter(X_test[variable_principal], y_test,
                    alpha=0.4, color='gray', s=20, label='Datos reales')
    # Curva predicha por el modelo
    y_curva = res['modelo'].predict(X_plot_df)
    axes[i].plot(X_plot[:, idx_bmi], y_curva,
                 color=color, linewidth=2.5, label='Curva del modelo')
    axes[i].set_xlabel(f'{variable_principal} (normalizado)', fontsize=11)
    axes[i].set_ylabel('Target (Progresión)', fontsize=11)
    axes[i].set_title(f'{titulo}', fontweight='bold')
    axes[i].legend(fontsize=9)

plt.tight_layout()
plt.savefig('modelado_02_curva_ajuste.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfica guardada: modelado_02_curva_ajuste.png")


# --- 10. GUARDAR EL MEJOR MODELO ---
# Identificamos cuál modelo tuvo mejor R² en test
mejor_res = max(resultados, key=lambda x: x['r2_test'])
mejor_modelo = mejor_res['modelo']
joblib.dump(mejor_modelo, 'mejor_modelo.pkl')
print(f"\n💾 Mejor modelo guardado: 'mejor_modelo.pkl'")
print(f"   → {mejor_res['nombre']}")
print(f"   → R² Test: {mejor_res['r2_test']:.4f} | RMSE Test: {mejor_res['rmse_test']:.2f}")

print("\n✅ Fase II (Modelado) completada.")
print("   Siguiente paso: Ejecutar fase3_evaluacion.py")
