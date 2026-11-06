# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 09:46:13 2026

@author: Usuario
"""


import numpy as np
from ForecastEngine import ForecastEngine
from ForecastTemporal import ForecastTemporal

# ==========================================================
# Reconstrucción del modelo entrenado
# ==========================================================

engine = ForecastEngine(
    earthquake_partition="IGP",
    Mcut=3.5,
    M0pred=4.5,
    time_step=20
)

engine.initialize()

# ==========================================================
# Pronóstico temporal
# ==========================================================

temporal = ForecastTemporal(
    engine.get_state(),
    engine.get_model()
)




temporal.build_time_grid()

print("\n==============================")
print("PRUEBA 1")
print("==============================")

print("Primer tiempo :", temporal.time_grid[0])
print("Último tiempo :", temporal.time_grid[-1])
print("Número puntos :", len(temporal.time_grid))

print("\n==============================")
print("PRUEBA 2")
print("==============================")

hazard, cumulative = temporal._predict_model(1.0)

print()

print("Tipo hazard      :", type(hazard))
print("Tipo cumulative  :", type(cumulative))

print()

print("Shape hazard     :", hazard.shape)
print("Shape cumulative :", cumulative.shape)

print()

print("Primer hazard    :", hazard[0])
print("Último hazard    :", hazard[-1])

print()

print("Primer cumulative :", cumulative[0])
print("Último cumulative :", cumulative[-1])

print()

print("==============================")

print("PRUEBA 3")

print("==============================")

density = temporal._density_from_outputs(hazard,cumulative)

print()

print("Density =", density)


temporal.build_time_grid(
    max_days=30,
    n_points=500
)

temporal.compute_density()

temporal.compute_cumulative()



# print()
# print("==============================")
# print("ANÁLISIS NaN")
# print("==============================")

# indices = np.where(np.isnan(temporal.temporal_density))[0]

# print("Número de NaN :", len(indices))

# print("Primeros índices :", indices[:20])


# print()

# print("==============================")
# print("PRUEBA 4")
# print("==============================")

# print("Shape :", temporal.temporal_density.shape)
# print("Primer valor :", temporal.temporal_density[0])
# print("Último valor :", temporal.temporal_density[-1])
# print("Máximo :", temporal.temporal_density.max())
# print("Mínimo :", temporal.temporal_density.min())






print()

print("==============================")

print("PRUEBA 5")

print("==============================")

print()

print("Shape :", temporal.temporal_cdf.shape)

print()

print("Primer valor :", temporal.temporal_cdf[0])

print("Último valor :", temporal.temporal_cdf[-1])

print()

print("P(T<=1 día)  :", temporal.probability_before(1))

print("P(T<=7 días) :", temporal.probability_before(7))

print("P(T<=30 días):", temporal.probability_before(30))


dt = temporal.time_grid[1] - temporal.time_grid[0]

area = np.trapz(
    temporal.temporal_density,
    temporal.time_grid
)

print()
print("==============================")
print("VALIDACIÓN")
print("==============================")
print("Área bajo la densidad (trapz):", area)





temporal.summary()