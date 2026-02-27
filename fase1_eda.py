# =============================================================================
# TALLER 1.1 - EQUIPO 3: DIABETES (Scikit-learn)
# FASE I: Comprensión de los Datos (Data Understanding / EDA)
# =============================================================================
# Objetivo: Analizar en profundidad el dataset de Diabetes para entender
# la distribución de variables, correlaciones y detectar outliers antes
# de construir cualquier modelo predictivo.
# =============================================================================

# --- 1. IMPORTACIÓN DE LIBRERÍAS ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.datasets import load_diabetes
import warnings
warnings.filterwarnings('ignore')

# Configuración global de gráficas
plt.rcParams['figure.facecolor'] = '#f8f9fa'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_palette("husl")

print("=" * 60)
print("  TALLER 1.1 | EQUIPO 3 | FASE I: EDA")
print("  Dataset: Diabetes (Scikit-learn)")
print("=" * 60)


# --- 2. CARGA DEL DATASET ---
diabetes = load_diabetes()  # Cargamos el dataset directamente desde sklearn

# Convertimos a DataFrame para facilitar el análisis
df = pd.DataFrame(
    data=diabetes.data,
    columns=diabetes.feature_names  # age, sex, bmi, bp, s1..s6
)

# Agregamos la variable target (lo que queremos predecir)
df['target'] = diabetes.target  # Progresión de diabetes al año siguiente

print("\n📋 DESCRIPCIÓN DEL DATASET:")
print(f"   → Muestras (pacientes): {df.shape[0]}")
print(f"   → Variables (features): {df.shape[1] - 1}")
print(f"   → Variable objetivo:    target (progresión de la enfermedad)")

# Descripción de cada variable
print("\n📌 VARIABLES DEL DATASET:")
variables_info = {
    'age':    'Edad del paciente (normalizada)',
    'sex':    'Sexo del paciente (normalizado)',
    'bmi':    'Índice de Masa Corporal (normalizado)',
    'bp':     'Presión arterial media (normalizada)',
    's1':     'Colesterol total en suero (normalizado)',
    's2':     'LDL (colesterol malo) (normalizado)',
    's3':     'HDL (colesterol bueno) (normalizado)',
    's4':     'Relación colesterol total / HDL (normalizado)',
    's5':     'Log de triglicéridos en suero (normalizado)',
    's6':     'Glucosa en sangre (normalizado)',
    'target': 'Progresión de diabetes 1 año después (variable objetivo)'
}
for var, desc in variables_info.items():
    print(f"   {var:8s}: {desc}")


# --- 3. ANÁLISIS ESTADÍSTICO BÁSICO ---
print("\n📊 ESTADÍSTICAS DESCRIPTIVAS:")
print(df.describe().round(4).to_string())

# Verificar valores nulos
print(f"\n🔍 VALORES NULOS: {df.isnull().sum().sum()} (ninguno)")

# Rango del target
print(f"\n🎯 RANGO DEL TARGET:")
print(f"   Mínimo: {df['target'].min():.1f}")
print(f"   Máximo: {df['target'].max():.1f}")
print(f"   Media:  {df['target'].mean():.1f}")
print(f"   Desv. Estándar: {df['target'].std():.1f}")


# --- 4. VISUALIZACIÓN: DISTRIBUCIÓN DEL TARGET ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Distribución de la Variable Objetivo (Target)',
             fontsize=14, fontweight='bold', y=1.02)

# Histograma
axes[0].hist(df['target'], bins=30, color='#4C72B0', edgecolor='white', alpha=0.85)
axes[0].axvline(df['target'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Media: {df['target'].mean():.1f}")
axes[0].axvline(df['target'].median(), color='orange', linestyle='--', linewidth=2,
                label=f"Mediana: {df['target'].median():.1f}")
axes[0].set_xlabel('Progresión de la Enfermedad')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Histograma del Target')
axes[0].legend()

# Boxplot
axes[1].boxplot(df['target'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='#4C72B0', alpha=0.7))
axes[1].set_ylabel('Progresión de la Enfermedad')
axes[1].set_title('Boxplot del Target')
axes[1].set_xticklabels(['target'])

plt.tight_layout()
plt.savefig('eda_01_distribucion_target.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfica guardada: eda_01_distribucion_target.png")


# --- 5. VISUALIZACIÓN: DISTRIBUCIÓN DE TODAS LAS FEATURES ---
features = diabetes.feature_names  # Lista de las 10 variables independientes

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Distribución de las Variables Independientes (Features)',
             fontsize=14, fontweight='bold')
axes = axes.flatten()

for i, col in enumerate(features):
    axes[i].hist(df[col], bins=25, color='#55A868', edgecolor='white', alpha=0.8)
    axes[i].axvline(df[col].mean(), color='red', linestyle='--', linewidth=1.5)
    axes[i].set_title(f'{col}', fontweight='bold')
    axes[i].set_xlabel('Valor (normalizado)')
    axes[i].set_ylabel('Frecuencia')

plt.tight_layout()
plt.savefig('eda_02_distribucion_features.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfica guardada: eda_02_distribucion_features.png")


# --- 6. MATRIZ DE CORRELACIÓN ---
# Esta es la visualización OBLIGATORIA del taller
# Nos muestra qué variables están más relacionadas con el target

corr_matrix = df.corr()  # Calculamos la correlación de Pearson entre todas las variables

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Análisis de Correlaciones', fontsize=14, fontweight='bold')

# Mapa de calor completo
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Máscara para triángulo superior
sns.heatmap(
    corr_matrix,
    annot=True,           # Mostrar valores numéricos
    fmt='.2f',            # 2 decimales
    cmap='RdYlGn',        # Verde=positivo, Rojo=negativo
    mask=mask,            # Solo triángulo inferior (evita redundancia)
    linewidths=0.5,
    ax=axes[0],
    vmin=-1, vmax=1,
    annot_kws={"size": 8}
)
axes[0].set_title('Matriz de Correlación Completa\n(Pearson)', fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)

# Correlaciones solo con el target (ranking)
corr_target = corr_matrix['target'].drop('target').sort_values(ascending=True)
colors = ['#d73027' if x < 0 else '#1a9850' for x in corr_target]
axes[1].barh(corr_target.index, corr_target.values, color=colors, edgecolor='white')
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].set_xlabel('Correlación con Target')
axes[1].set_title('Correlación de cada Variable con el Target\n(ordenado)', fontweight='bold')
for i, (val, name) in enumerate(zip(corr_target.values, corr_target.index)):
    axes[1].text(val + (0.005 if val >= 0 else -0.005), i,
                 f'{val:.3f}', va='center',
                 ha='left' if val >= 0 else 'right', fontsize=9)

plt.tight_layout()
plt.savefig('eda_03_correlaciones.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfica guardada: eda_03_correlaciones.png")

# Interpretación de correlaciones
print("\n📈 CORRELACIONES CON EL TARGET (ordenadas):")
print(corr_target.sort_values(ascending=False).to_string())
print("\n💡 INTERPRETACIÓN:")
print("   Variables con MAYOR correlación positiva (más influyentes):")
top3 = corr_target.sort_values(ascending=False).head(3)
for var, val in top3.items():
    print(f"     → {var}: {val:.3f}")
print("\n   Variables con correlación NEGATIVA:")
neg = corr_target[corr_target < 0]
for var, val in neg.items():
    print(f"     → {var}: {val:.3f} (inversa al target)")


# --- 7. SCATTER PLOTS: VARIABLES MÁS RELEVANTES VS TARGET ---
# Mostramos las 6 variables con mayor correlación absoluta
top6_vars = corr_target.abs().sort_values(ascending=False).head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Relación de las 6 Variables más Correlacionadas con el Target',
             fontsize=13, fontweight='bold')
axes = axes.flatten()

for i, col in enumerate(top6_vars):
    axes[i].scatter(df[col], df['target'], alpha=0.4, color='#4C72B0', s=20)
    # Línea de tendencia (regresión lineal simple)
    z = np.polyfit(df[col], df['target'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[col].min(), df[col].max(), 100)
    axes[i].plot(x_line, p(x_line), "r--", linewidth=2, label='Tendencia')
    axes[i].set_xlabel(f'{col} (normalizado)', fontsize=10)
    axes[i].set_ylabel('Target (Progresión)', fontsize=10)
    corr_val = corr_matrix.loc[col, 'target']
    axes[i].set_title(f'{col} vs Target\n(r = {corr_val:.3f})', fontweight='bold')
    axes[i].legend(fontsize=8)

plt.tight_layout()
plt.savefig('eda_04_scatter_top6.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfica guardada: eda_04_scatter_top6.png")


# --- 8. DETECCIÓN DE OUTLIERS CON BOXPLOTS ---
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Detección de Outliers por Variable (Boxplots)', fontsize=13, fontweight='bold')
axes = axes.flatten()

for i, col in enumerate(features):
    bp = axes[i].boxplot(df[col], patch_artist=True,
                         boxprops=dict(facecolor='#4C72B0', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2))
    axes[i].set_title(col, fontweight='bold')
    axes[i].set_ylabel('Valor normalizado')
    # Contar outliers
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    axes[i].set_xlabel(f'{len(outliers)} outliers', fontsize=9, color='red')

plt.tight_layout()
plt.savefig('eda_05_outliers.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfica guardada: eda_05_outliers.png")


# --- 9. ANÁLISIS DE MULTICOLINEALIDAD ---
# Variables altamente correlacionadas entre sí pueden causar problemas al modelo

print("\n⚠️  ANÁLISIS DE MULTICOLINEALIDAD:")
print("   (Pares de features con correlación > 0.70 o < -0.70)")
threshold = 0.70
high_corr_pairs = []
feature_corr = df[features].corr()
for i in range(len(features)):
    for j in range(i+1, len(features)):
        val = feature_corr.iloc[i, j]
        if abs(val) >= threshold:
            high_corr_pairs.append((features[i], features[j], val))
            print(f"   → {features[i]} ↔ {features[j]}: {val:.3f}")

if not high_corr_pairs:
    print("   No se encontraron pares con correlación mayor a 0.70")


# --- 10. RESUMEN FINAL DEL EDA ---
print("\n" + "=" * 60)
print("  📝 RESUMEN EDA - HALLAZGOS PRINCIPALES")
print("=" * 60)
print(f"""
1. El dataset contiene 442 muestras y 10 variables predictoras.
   Todas las variables ya están NORMALIZADAS (media=0, std=1),
   por lo que NO necesitamos escalar los datos.

2. VARIABLES MÁS INFLUYENTES en el target (por correlación):
   → bmi (IMC): correlación más alta positiva
   → s5 (log triglicéridos): fuerte correlación positiva
   → bp (presión arterial): correlación positiva moderada
   → s3 (HDL): correlación negativa (a más HDL, menos progresión)

3. OUTLIERS: Presentes en algunas variables pero en cantidad
   normal; no se recomienda eliminarlos para no perder datos médicos.

4. MULTICOLINEALIDAD: Algunas variables de lípidos (s1-s4) están
   correlacionadas entre sí. Esto puede afectar la interpretación
   de coeficientes en el modelo lineal.

5. TARGET: Distribución aproximadamente normal (centrada ~150),
   rango de 25 a 346. Adecuado para regresión lineal.
""")
print("✅ Fase I (EDA) completada exitosamente.")
print("   Siguiente paso: Ejecutar fase2_modelado.py")
