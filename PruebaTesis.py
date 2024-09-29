from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Datos de los dos grupos
datos_grupo_a = [7.2, 6.6, 7.4, 6, 6.4, 7.1, 10,7.3]
datos_grupo_b = [7, 7.4, 6.9, 4, 7.6, 7.7, 5, 6.1]

desviacion_estandar_grupo_a = np.std(datos_grupo_a)
desviacion_estandar_grupo_b = np.std(datos_grupo_b)

promedio_grupo_a=np.average(datos_grupo_a)
promedio_grupo_b=np.average(datos_grupo_b)
print(f"Desviación estándar Grupo A: {desviacion_estandar_grupo_a}")
print(f"Promedio Grupo A: {promedio_grupo_a}")
print(f"Desviación estándar Grupo B: {desviacion_estandar_grupo_b}")
print(f"Promedio Grupo B: {promedio_grupo_b}")

# Realizar la prueba t
resultado_t, valor_p = stats.ttest_ind(datos_grupo_a, datos_grupo_b)
# Obtener los grados de libertad
grados_libertad = len(datos_grupo_a) + len(datos_grupo_b) - 2 
# Imprimir el resultado
print("Valor t:", resultado_t)
print("Valor p:", valor_p)
print("Grados de libertad:", grados_libertad)
# Concatenar datos de ambos grupos
todos_los_datos = datos_grupo_a + datos_grupo_b
# Crear un DataFrame con los datos y etiquetas de grupo
import pandas as pd
df = pd.DataFrame({
    'Valor': todos_los_datos,
    'Grupo': ['Grupo A'] * len(datos_grupo_a) + ['Grupo B'] * len(datos_grupo_b)
})
# Crear la gráfica de densidad
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='Valor', hue='Grupo', fill=True, common_norm=False)
plt.title('Distribución de Datos entre Grupo A y Grupo B')
plt.show()