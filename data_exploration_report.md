# Exploración de datos - Bike Rental Daily

## Dimensiones del conjunto
- Filas: 600
- Columnas: 18

## Valores faltantes (ANTES de la limpieza)
- `season`: 62 valores faltantes
- `hum`: 34 valores faltantes
- Resto de columnas: 0 valores faltantes

## Acciones de limpieza de datos realizadas
- **season**: 62 valores rellenados con la moda (1 = invierno)
- **hum**: 34 valores rellenados con la media (93.85%)
- **atemp**: Recalculado para todas las filas usando la fórmula de Heat Index (sensación térmica) del NOAA
- **weekday**: 17 registros con valor -1 corregidos a 0
- **Resultado**: Dataset completamente limpio sin valores faltantes

## Variables presentes
- instant, dteday, season, yr, mnth, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed, leaflets, price reduction, casual, registered, cnt

## Valores atípicos y problemas de calidad
- `weekday` contiene 17 registros con valor `-1`, lo cual es inconsistente con la escala esperada 0-6.
- `cnt` tiene valores muy altos en algunos registros: 53021, 42945, 35932, 32043.
- `casual` también muestra valores extremos claramente asociados a esos días de alta demanda.
- `weathersit` jamás aparece con valor 4 en este conjunto de entrenamiento.

## Estadísticas descriptivas clave
- `cnt` promedio: 4705
- `cnt` mediana: 4530
- `cnt` máximo: 53021
- `temp` promedio: 19.8
- `hum` promedio: 93.8
- `windspeed` promedio: 0.184
- `price reduction` es un evento raro: 90 días con reducción frente a 510 días sin reducción.

## Visualizaciones generadas
- `outputs/chart_cnt_hist.png`: histograma de `cnt`
- `outputs/chart_numeric_box.png`: boxplot de variables numéricas (`temp`, `hum`, `windspeed`, `leaflets`)
- `outputs/chart_cnt_box.png`: boxplot de `cnt`
- `outputs/chart_cnt_year.png`: promedio de `cnt` por año
- `outputs/chart_cnt_weather.png`: promedio de `cnt` por estado del clima

## Insights principales
- La distribución de `cnt` es asimétrica y presenta valores extremos que pueden ser outliers o días de demanda extraordinaria.
- `season`, `yr`, `mnth`, `holiday`, `weekday`, `workingday` y `weathersit` son claramente variables categóricas.
- `temp`, `hum`, `windspeed` y `leaflets` deben normalizarse o estandarizarse antes de aplicar muchos modelos de regresión.
- `price reduction` es una variable binaria útil, aunque desbalanceada.
- `casual` y `registered` no deben usarse como características cuando `cnt` es el target, porque representan directamente su desagregación.

## Recomendaciones para la siguiente fase
- Corregir o imputar los valores faltantes en `season` y `hum`.
- Revisar y ajustar `weekday=-1` para hacerlo consistente o excluir esos registros si no se puede corregir.
- Convertir `dteday` a un tipo fecha y posiblemente extraer información adicional (día, mes, trimestre, fin de semana, feriado).
- Transformar variables numéricas para escalarlas (scaling) y/o aplicar codificación categórica a variables como `season`, `weathersit`.
- No eliminar filas de validación o prueba; sólo tratar los valores faltantes y variables problemáticas.
- Evaluar modelos base con regresión lineal, árbol de decisión, SVR y k-NN, y comparar su desempeño.
