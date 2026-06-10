#!/usr/bin/env python3
"""
Script para completar datos faltantes usando heat index (sensación térmica).
"""

import pandas as pd
import numpy as np
from pathlib import Path

def heat_index_celsius(T, RH):
    """
    Calcula el Heat Index (sensación térmica) en °C
    usando la fórmula oficial del NOAA.
    T = temperatura en °C
    RH = humedad relativa en %
    """
    # Convertir a Fahrenheit
    T_F = T * 1.8 + 32
    
    # Fórmula del Heat Index en °F
    HI_F = (-42.379 + 2.04901523*T_F + 10.14333127*RH
            - 0.22475541*T_F*RH - 6.83783e-3*T_F**2
            - 5.481717e-2*RH**2 + 1.22874e-3*T_F**2*RH
            + 8.5282e-4*T_F*RH**2 - 1.99e-6*T_F**2*RH**2)
    
    # Convertir de vuelta a °C
    HI_C = (HI_F - 32) / 1.8
    return HI_C

def calc_atemp(T, RH):
    """
    Calcula la sensación térmica normalizada (0–1)
    como en el dataset Bike Sharing.
    """
    HI = heat_index_celsius(T, RH)
    return max(0, min(1, HI / 50))  # Normalización y recorte a 0–1

# Leer el CSV
csv_path = Path('data/BikeRentalDaily_train.csv')
df = pd.read_csv(csv_path, sep=';')

print(f"Dataset original: {df.shape}")
print(f"Valores faltantes antes:")
print(df[['temp', 'hum', 'atemp', 'season']].isna().sum())

# 1. Rellenar season con la moda (valor más frecuente)
if df['season'].isna().sum() > 0:
    mode_season = df['season'].mode()[0]
    print(f"\nRellenando season (faltantes: {df['season'].isna().sum()}) con moda: {mode_season}")
    df['season'].fillna(mode_season, inplace=True)

# 2. Rellenar hum con la media
if df['hum'].isna().sum() > 0:
    mean_hum = df['hum'].mean()
    print(f"Rellenando hum (faltantes: {df['hum'].isna().sum()}) con media: {mean_hum:.2f}")
    df['hum'].fillna(mean_hum, inplace=True)

# 3. Recalcular atemp usando la fórmula de heat index para todas las filas
print("\nRecalculando atemp usando heat index...")
df['atemp'] = df.apply(lambda row: calc_atemp(row['temp'], row['hum']), axis=1)

# 4. Corregir weekday=-1 (si hay)
if (df['weekday'] == -1).sum() > 0:
    print(f"\nEncontrados {(df['weekday'] == -1).sum()} registros con weekday=-1")
    print("Asignando el día de la semana más frecuente (0) a estos registros...")
    df.loc[df['weekday'] == -1, 'weekday'] = 0

print(f"\nValores faltantes después:")
print(df[['temp', 'hum', 'atemp', 'season']].isna().sum())

# Guardar el CSV limpio
output_path = Path('data/BikeRentalDaily_train_clean.csv')
df.to_csv(output_path, sep=';', index=False)
print(f"\nDataset limpio guardado en: {output_path}")
print(f"Nueva forma: {df.shape}")

# Mostrar estadísticas del dataset limpio
print("\nEstadísticas del dataset limpio:")
print(df[['temp', 'hum', 'atemp']].describe().transpose())
