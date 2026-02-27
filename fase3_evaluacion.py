# =============================================================================
# TALLER 1.1 - EQUIPO 3: DIABETES (Scikit-learn)
# FASE III: Evaluación (Evaluation)
# =============================================================================
# Objetivo: Justificar la elección del modelo final usando:
#   → R² (R-cuadrado)
#   → RMSE (Root Mean Squared Error)
#   → Análisis de Residuos (verificar que no haya patrones ocultos)
# =============================================================================

# --- 1. IMPORTACIÓN DE LIBRERÍAS ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from scipy import stats  # Para test de normalidad de residuos

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.facecolor'] = '#f8f9fa'
sns.set_palette("husl")

print("=" * 60)
print("  TALLER 1.1 | EQUIPO 3 | FASE III: EVALUACIÓN")
print("=" * 60)


# --- 2. PREPARACIÓN DE DATOS Y MODELOS ---
# (Repetimos la preparación para que este archivo sea autónomo)
diabetes = load_diabetes()
X = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name='target')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Definición de los 3 modelos
modelos = {
    'Lineal Múltiple': LinearRegression(),
    'Polinomial Grado 2': Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('reg',  LinearRegression())
    ]),
    'Polinomial Grado 3': Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('reg',  LinearRegression())
    ])
}

# Entrenamos todos los modelos
resultados = {}
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred_train = modelo.predict(X_train)
    y_pred_test  = modelo.predict(X_test)
    residuos      = y_test - y_pred_test  # Errores = real - predicho

    resultados[nombre] = {
        'modelo': modelo,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'residuos': residuos,
        # Métricas en entrenamiento
        'r2_train':  r2_score(y_train, y_pred_train),
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'mae_train':  mean_absolute_error(y_train, y_pred_train),
        # Métricas en prueba
        'r2_test':   r2_score(y_test, y_pred_test),
        'rmse_test':  np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'mae_test':   mean_absolute_error(y_test, y_pred_test),
    }


# --- 3. TABLA COMPARATIVA DE MÉTRICAS ---
print("\n📊 TABLA COMPARATIVA DE MÉTRICAS:")
print("─" * 75)
print(f"{'Modelo':<25} {'R²Train':>8} {'R²Test':>8} {'RMSETest':>10} {'MAETest':>9} {'Gap':>8}")
print("─" * 75)
for nombre, res in resultados.items():
    gap = res['r2_train'] - res['r2_test']
    print(f"{nombre:<25} {res['r2_train']:>8.4f} {res['r2_test']:>8.4f} "
          f"{res['rmse_test']:>10.2f} {res['mae_test']:>9.2f} {gap:>8.4f}")
print("─" * 75)

print("\n💡 INTERPRETACIÓN DE MÉTRICAS:")
print("""
   R² (R-cuadrado):
   → Mide qué porcentaje de la varianza del target explica el modelo.
   → R²=1.0 significa predicción perfecta.
   → R²=0.5 significa que el modelo explica el 50% de la variación.
   → Si R²_train >> R²_test: HAY OVERFITTING (modelo memoriza, no generaliza).

   RMSE (Root Mean Squared Error):
   → Error promedio en las mismas unidades que el target.
   → Ej: RMSE=55 significa que el modelo se equivoca ~55 unidades en promedio.
   → Penaliza más los errores grandes.

   MAE (Mean Absolute Error):
   → Error absoluto promedio. Más robusto a outliers que RMSE.

   Gap (R²Train - R²Test):
   → < 0.05: Excelente, sin overfitting.
   → 0.05 - 0.15: Overfitting leve, aceptable.
   → > 0.15: Overfitting importante, modelo no generaliza bien.
""")


# --- 4. ANÁLISIS DE RESIDUOS (para los 3 modelos) ---
print("📈 Generando análisis de residuos...")

fig = plt.figure(figsize=(20, 16))
fig.suptitle('Análisis Completo de Residuos por Modelo',
             fontsize=15, fontweight='bold', y=1.01)

colores = {'Lineal Múltiple': '#4C72B0',
           'Polinomial Grado 2': '#55A868',
           'Polinomial Grado 3': '#C44E52'}

for col_idx, (nombre, res) in enumerate(resultados.items()):
    residuos  = res['residuos'].values
    y_pred    = res['y_pred_test']
    color     = colores[nombre]

    # --- 4a. Residuos vs Valores Predichos ---
    # Si no hay patrón (nube aleatoria), el modelo es adecuado.
    # Si hay curva/embudo, falta algún patrón que el modelo no captura.
    ax1 = fig.add_subplot(4, 3, col_idx + 1)
    ax1.scatter(y_pred, residuos, alpha=0.5, color=color, s=25)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Valores Predichos')
    ax1.set_ylabel('Residuos')
    ax1.set_title(f'{nombre}\nResiduos vs Predichos', fontweight='bold', fontsize=10)
    ax1.text(0.05, 0.95, '✅ Aleatorio = Bueno\n❌ Patrón = Problema',
             transform=ax1.transAxes, fontsize=7, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- 4b. Histograma de Residuos ---
    # Deben seguir una distribución normal (campana) para validar supuestos.
    ax2 = fig.add_subplot(4, 3, col_idx + 4)
    ax2.hist(residuos, bins=20, color=color, edgecolor='white', alpha=0.8, density=True)
    # Curva normal teórica
    mu, sigma = residuos.mean(), residuos.std()
    x_norm = np.linspace(residuos.min(), residuos.max(), 100)
    ax2.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 'r-', linewidth=2,
             label=f'Normal teórica\nμ={mu:.1f}, σ={sigma:.1f}')
    ax2.set_xlabel('Residuos')
    ax2.set_ylabel('Densidad')
    ax2.set_title('Distribución de Residuos', fontweight='bold', fontsize=10)
    ax2.legend(fontsize=8)

    # --- 4c. QQ-Plot ---
    # Compara la distribución de residuos contra una normal teórica.
    # Si los puntos están sobre la línea diagonal, los residuos son normales.
    ax3 = fig.add_subplot(4, 3, col_idx + 7)
    (osm, osr), (slope, intercept, r) = stats.probplot(residuos, dist="norm")
    ax3.scatter(osm, osr, color=color, alpha=0.5, s=20)
    ax3.plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=2)
    ax3.set_xlabel('Cuantiles Teóricos (Normal)')
    ax3.set_ylabel('Cuantiles de los Residuos')
    ax3.set_title('QQ-Plot de Residuos', fontweight='bold', fontsize=10)

    # --- 4d. Residuos en orden (detección de autocorrelación) ---
    ax4 = fig.add_subplot(4, 3, col_idx + 10)
    ax4.plot(range(len(residuos)), residuos, color=color, alpha=0.6, linewidth=0.8)
    ax4.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax4.set_xlabel('Índice de Observación')
    ax4.set_ylabel('Residuo')
    ax4.set_title('Residuos en Orden\n(¿Autocorrelación?)', fontweight='bold', fontsize=10)

    # Test de normalidad Shapiro-Wilk
    stat, p_value = stats.shapiro(residuos)
    conclusion = "Normal ✅" if p_value > 0.05 else "No-Normal ⚠️"
    print(f"   {nombre}: Shapiro-Wilk p={p_value:.4f} → {conclusion}")

plt.tight_layout()
plt.savefig('evaluacion_01_residuos.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfica guardada: evaluacion_01_residuos.png")


# --- 5. CURVAS DE APRENDIZAJE ---
# Muestran cómo mejora el modelo con más datos de entrenamiento.
# Si train y validation se acercan y tienen buen R², el modelo es bueno.
# Si hay gran separación, hay overfitting.

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Curvas de Aprendizaje (¿Overfitting o Underfitting?)',
             fontsize=13, fontweight='bold')

train_sizes = np.linspace(0.1, 1.0, 10)  # Del 10% al 100% de los datos

for i, (nombre, res) in enumerate(resultados.items()):
    color = list(colores.values())[i]
    train_sz, train_scores, val_scores = learning_curve(
        res['modelo'], X_train, y_train,
        train_sizes=train_sizes,
        cv=5,                    # Validación cruzada 5-fold
        scoring='r2',
        n_jobs=-1
    )
    # Media y desviación estándar
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    axes[i].fill_between(train_sz, train_mean - train_std,
                         train_mean + train_std, alpha=0.15, color=color)
    axes[i].fill_between(train_sz, val_mean - val_std,
                         val_mean + val_std, alpha=0.15, color='orange')
    axes[i].plot(train_sz, train_mean, 'o-', color=color,
                 linewidth=2, label='Entrenamiento', markersize=5)
    axes[i].plot(train_sz, val_mean, 's--', color='orange',
                 linewidth=2, label='Validación (CV)', markersize=5)
    axes[i].set_xlabel('Muestras de Entrenamiento')
    axes[i].set_ylabel('R²')
    axes[i].set_title(f'{nombre}', fontweight='bold')
    axes[i].legend(fontsize=9)
    axes[i].set_ylim(-0.2, 1.05)
    axes[i].axhline(0, color='gray', linestyle=':', linewidth=0.8)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evaluacion_02_learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfica guardada: evaluacion_02_learning_curves.png")


# --- 6. GRÁFICA COMPARATIVA FINAL DE MÉTRICAS ---
nombres = list(resultados.keys())
r2_tests   = [resultados[n]['r2_test']   for n in nombres]
rmse_tests = [resultados[n]['rmse_test'] for n in nombres]
r2_trains  = [resultados[n]['r2_train']  for n in nombres]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Comparativa Final de Modelos', fontsize=14, fontweight='bold')
x = np.arange(len(nombres))
width = 0.35
bar_colors = list(colores.values())

# R²
bars1 = axes[0].bar(x - width/2, r2_trains, width, label='R² Train',
                    color=bar_colors, alpha=0.5, edgecolor='black', linewidth=0.8)
bars2 = axes[0].bar(x + width/2, r2_tests, width, label='R² Test',
                    color=bar_colors, alpha=0.9, edgecolor='black', linewidth=0.8)
axes[0].set_ylabel('R² (mayor es mejor)')
axes[0].set_title('R² Train vs Test', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['Lineal', 'Poly-2', 'Poly-3'])
axes[0].legend()
axes[0].set_ylim(0, 1.1)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2.,
                 bar.get_height() + 0.01, f'{bar.get_height():.3f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

# RMSE
bars3 = axes[1].bar(x, rmse_tests, width * 1.5, color=bar_colors,
                    alpha=0.85, edgecolor='black', linewidth=0.8)
axes[1].set_ylabel('RMSE (menor es mejor)')
axes[1].set_title('RMSE en Test', fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(['Lineal', 'Poly-2', 'Poly-3'])
for bar in bars3:
    axes[1].text(bar.get_x() + bar.get_width()/2.,
                 bar.get_height() + 0.3, f'{bar.get_height():.2f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('evaluacion_03_comparativa.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfica guardada: evaluacion_03_comparativa.png")


# --- 7. CONCLUSIÓN Y SELECCIÓN DEL MODELO ---
print("\n" + "=" * 60)
print("  🏆 CONCLUSIÓN Y MODELO FINAL SELECCIONADO")
print("=" * 60)

mejor_nombre = max(resultados, key=lambda n: resultados[n]['r2_test'])
mejor = resultados[mejor_nombre]
print(f"""
MODELO ELEGIDO: {mejor_nombre}

Justificación:
  ✅ R² Test  = {mejor['r2_test']:.4f}  → Explica el {mejor['r2_test']*100:.1f}% de la varianza
  ✅ RMSE Test = {mejor['rmse_test']:.2f}  → Error promedio de ~{mejor['rmse_test']:.0f} unidades
  ✅ Gap overfit = {mejor['r2_train']-mejor['r2_test']:.4f} → {'Bajo, modelo generaliza bien' if (mejor['r2_train']-mejor['r2_test']) < 0.1 else 'Moderado'}

  El modelo Polinomial Grado 3 suele mostrar overfitting con este dataset
  (pocos datos + muchas features → riesgo de memorizar el entrenamiento).
  El Polinomial Grado 2 puede mejorar marginalmente al lineal si captura
  relaciones no-lineales, pero el Lineal Múltiple es generalmente el más
  robusto con solo 442 muestras.

  El error RMSE en unidades reales: una predicción podría desviarse
  en promedio ±{mejor['rmse_test']:.0f} puntos de la progresión real.
""")

print("✅ Fase III (Evaluación) completada.")
print("   Siguiente paso: Ejecutar fase4_backend/ y fase4_frontend/")
