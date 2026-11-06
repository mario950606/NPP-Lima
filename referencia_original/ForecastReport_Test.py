# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 19:01:41 2026

@author: Usuario
"""

from ForecastEngine import ForecastEngine
from ForecastTemporal import ForecastTemporal
from ForecastMagnitude import ForecastMagnitude
from ForecastReport import ForecastReport

##############################################################################
# Engine
##############################################################################

engine = ForecastEngine(
    earthquake_partition="IGP",
    Mcut=3.5,
    M0pred=4.5,
    time_step=20
)

engine.initialize()

##############################################################################
# Temporal
##############################################################################

temporal = ForecastTemporal(

    engine.get_state(),

    engine.get_model()

)

##############################################################################
# Construir la malla temporal
##############################################################################

temporal.build_time_grid()

##############################################################################
# Calcular densidad y CDF
##############################################################################

temporal.compute_density()

temporal.compute_cumulative()
##############################################################################
# Magnitud
##############################################################################

magnitude = ForecastMagnitude(

    engine.get_state(),

    engine.get_model()

)

magnitude.compute_density()

magnitude.compute_cumulative()

##############################################################################
# Reporte
##############################################################################

report = ForecastReport(

    temporal,

    magnitude

)

report.summary()

report.build_report()

print()

print("="*70)
print("REPORTE")
print("="*70)

report.print_report()

# for key, value in report.report.items():

#     print("{:<20} {}".format(key, value))