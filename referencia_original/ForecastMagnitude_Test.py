# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:14:59 2026

@author: Usuario
"""

# -*- coding: utf-8 -*-

from ForecastEngine import ForecastEngine
from ForecastMagnitude import ForecastMagnitude
import numpy as np

###############################################################################
# Reconstrucción del modelo
###############################################################################

engine = ForecastEngine(
    earthquake_partition="IGP",
    Mcut=3.5,
    M0pred=4.5,
    time_step=20
)

engine.initialize()

###############################################################################
# Obtener estado y modelo
###############################################################################

state = engine.get_state()

model = engine.get_model()

###############################################################################
# Crear ForecastMagnitude
###############################################################################

forecast = ForecastMagnitude(
    state,
    model
)

###############################################################################
# Mostrar información
###############################################################################

forecast.summary()

###############################################################################
# Construir malla de magnitudes
###############################################################################

forecast.build_magnitude_grid()

print()
print("=" * 70)
print("Primeras 10 magnitudes")
print("=" * 70)
print(forecast.magnitude_grid[:10])

print()
print("=" * 70)
print("Últimas 10 magnitudes")
print("=" * 70)
print(forecast.magnitude_grid[-10:])



hazard, cumulative = forecast._predict_model(3.5)

print()
print("="*70)
print("PRUEBA PREDICT")
print("="*70)

print("Hazard :", hazard[-1])
print("Cumulative :", cumulative[-1])


###############################################################################
# PRUEBA DE DENSIDAD
###############################################################################

forecast.compute_density()

print()

print("=" * 70)
print("Primeros 10 valores de la densidad")
print("=" * 70)

print(forecast.magnitude_density[:10])

print()

print("=" * 70)
print("Últimos 10 valores de la densidad")
print("=" * 70)

print(forecast.magnitude_density[-10:])

print()

print("=" * 70)
print("VALIDACIÓN")
print("=" * 70)

print("Área bajo la curva :")

print(np.trapz(

    forecast.magnitude_density,

    forecast.magnitude_grid

))

print()
print("="*70)
print("DIAGNÓSTICO DE LOS PRIMEROS 15 PUNTOS")
print("="*70)

for i in range(15):

    m = forecast.magnitude_grid[i]

    hazard, cumulative = forecast._predict_model(m)

    density = forecast._density_from_outputs(
        hazard,
        cumulative
    )

    print(
        "{:2d}  M={:.6f}   Hazard={:.6f}   Cum={:.6f}   Density={:.6f}".format(
            i,
            m,
            float(hazard[-1]),
            float(cumulative[-1]),
            density
        )
    )

###############################################################################
# PRUEBA CDF
###############################################################################

forecast.compute_cumulative()

print()

print("="*70)
print("Primeros 10 valores CDF")
print("="*70)

print(forecast.magnitude_cdf[:10])

print()

print("="*70)
print("Últimos 10 valores CDF")
print("="*70)

print(forecast.magnitude_cdf[-10:])

###############################################################################
# PRUEBA PROBABILIDAD
###############################################################################

print()

print("=" * 70)
print("PROBABILIDADES")
print("=" * 70)

print("P(M >= 3.0) :", forecast.probability_above(3.0))
print("P(M >= 3.5) :", forecast.probability_above(3.5))
print("P(M >= 4.0) :", forecast.probability_above(4.0))
print("P(M >= 5.0) :", forecast.probability_above(5.0))
print("P(M >= 6.0) :", forecast.probability_above(6.0))


