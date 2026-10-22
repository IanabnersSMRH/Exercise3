import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

# Configuración de estilo para gráficos
sns.set_theme(style="whitegrid")

# Generación de dataset con ruido sintético
data = {
    'edad': [25, 30, np.nan, 45, 22, 120, 35, 28, np.nan, 40], # 120 es atípico/erróneo
    'ingreso_mensual': [15000, 22000, 18000, np.nan, 12000, 950000, 25000, 21000, 19000, np.nan], # 950k outlier
    'scoring_credito': [650, 700, 680, 710, 590, 800, 670, 640, 620, 730]
}
df_dirty = pd.DataFrame(data)

print("=== DATASET ORIGINAL (CON RUIDO) ===")
print(df_dirty)

# Visualización previa (Opcional para el reporte: Boxplot de ingresos y edad antes de limpiar)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df_dirty['ingreso_mensual'], color='salmon')
plt.title('Boxplot Ingreso Mensual (Con Outliers)')

plt.subplot(1, 2, 2)
sns.boxplot(y=df_dirty['edad'], color='skyblue')
plt.title('Boxplot Edad (Con Errores)')
plt.tight_layout()
plt.savefig('boxplot_antes_limpieza.png')
plt.close()

# 1. Tratamiento de Valores Atípicos por IQR sobre 'ingreso_mensual'
Q1 = df_dirty['ingreso_mensual'].quantile(0.25)
Q3 = df_dirty['ingreso_mensual'].quantile(0.75)
IQR = Q3 - Q1
limite_superior = Q3 + 1.5 * IQR

print(f"\nQ1 (Ingreso): {Q1:,.2f} | Q3 (Ingreso): {Q3:,.2f} | IQR: {IQR:,.2f}")
print(f"Límite Superior para Outliers en Ingreso: ${limite_superior:,.2f}")

# Reemplazar outliers extremos con NaN para tratarlos en la imputación
df_dirty.loc[df_dirty['ingreso_mensual'] > limite_superior, 'ingreso_mensual'] = np.nan
df_dirty.loc[df_dirty['edad'] > 100, 'edad'] = np.nan  # Corrección de valor erróneo (Edad 120)

# 2. Imputación Múltiple Avanzada con KNNImputer (k=2)
imputer = KNNImputer(n_neighbors=2)
df_clean_array = imputer.fit_transform(df_dirty)
df_clean = pd.DataFrame(df_clean_array, columns=df_dirty.columns)

print("\n=== DATASET LIMPIO E IMPUTADO (KNN k=2) ===")
print(df_clean)

# Visualización posterior (Boxplot después de imputación)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df_clean['ingreso_mensual'], color='lightgreen')
plt.title('Boxplot Ingreso Mensual (Limpio e Imputado)')

plt.subplot(1, 2, 2)
sns.boxplot(y=df_clean['edad'], color='lightgreen')
plt.title('Boxplot Edad (Limpio e Imputado)')
plt.tight_layout()
plt.savefig('boxplot_despues_limpieza.png')
plt.close()

# 3. Experimentación con k=3 para comparar
imputer_k3 = KNNImputer(n_neighbors=3)
df_clean_k3 = pd.DataFrame(imputer_k3.fit_transform(df_dirty), columns=df_dirty.columns)

print("\n=== DATASET IMPUTADO CON (KNN k=3) ===")
print(df_clean_k3)