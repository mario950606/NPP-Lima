# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 19:00:11 2026

@author: Usuario
"""

# -*- coding: utf-8 -*-
"""
ForecastReport.py

Generación del reporte final del pronóstico
Neural Point Process.

Este módulo NO realiza inferencia.

Únicamente organiza y resume los resultados
obtenidos por:

    ForecastTemporal
    ForecastMagnitude
"""

import numpy as np


class ForecastReport(object):

    """
    Reporte integrado del pronóstico.
    """

    #####################################################################
    # Constructor
    #####################################################################

    def __init__(

            self,

            temporal,

            magnitude):

        self.temporal = temporal

        self.magnitude = magnitude

        ############################################################

        self._validate_inputs()

        self._initialize_results()

    #####################################################################
    # Validación
    #####################################################################

    def _validate_inputs(self):

        if self.temporal is None:

            raise RuntimeError(

                "ForecastTemporal no disponible."

            )

        if self.magnitude is None:

            raise RuntimeError(

                "ForecastMagnitude no disponible."

            )

        ############################################################

        if self.temporal.temporal_density is None:

            raise RuntimeError(

                "Debe ejecutar compute_density() en ForecastTemporal."

            )

        if self.temporal.temporal_cdf is None:

            raise RuntimeError(

                "Debe ejecutar compute_cumulative() en ForecastTemporal."

            )

        ############################################################

        if self.magnitude.magnitude_density is None:

            raise RuntimeError(

                "Debe ejecutar compute_density() en ForecastMagnitude."

            )

        if self.magnitude.magnitude_cdf is None:

            raise RuntimeError(

                "Debe ejecutar compute_cumulative() en ForecastMagnitude."

            )

    #####################################################################
    # Inicialización
    #####################################################################

    def _initialize_results(self):
        
        ############################################################
        # Diccionario donde se almacenará el reporte
        ############################################################

        
        self.report = {}

        self.report_ready = False

    #####################################################################
    # Resumen
    #####################################################################

    def summary(self):

        print()

        print("="*70)

        print("FORECAST REPORT")

        print("="*70)

        print()

        print("ForecastTemporal :", self.temporal is not None)

        print("ForecastMagnitude:", self.magnitude is not None)

        print()

        print("Temporal")

        print("------------------------------")

        print("Grid      :", self.temporal.time_grid.shape)

        print("Density   :", self.temporal.temporal_density.shape)

        print("CDF       :", self.temporal.temporal_cdf.shape)

        print()

        print("Magnitude")

        print("------------------------------")

        print("Grid      :", self.magnitude.magnitude_grid.shape)

        print("Density   :", self.magnitude.magnitude_density.shape)

        print("CDF       :", self.magnitude.magnitude_cdf.shape)

        print()

        print("Estado del reporte :", self.report_ready)

        print()

        print("="*70)
        
        ##############################################################################
        # Construcción del reporte
        ##############################################################################

    def build_report(self):

        """
        Construye el reporte completo utilizando los resultados
        calculados por ForecastTemporal y ForecastMagnitude.
        """

        ############################################################
        # Último evento
        ############################################################

        state = self.temporal.state

        self.report["last_time"] = state.last_time

        self.report["last_magnitude"] = state.last_magnitude

        ############################################################
        # Temporal
        ############################################################

        self.report["P_1_day"] = self.temporal.probability_before(1)

        self.report["P_7_days"] = self.temporal.probability_before(7)

        self.report["P_30_days"] = self.temporal.probability_before(30)

        ############################################################
        # Magnitud
        ############################################################

        self.report["P_M3"] = self.magnitude.probability_above(3.0)

        self.report["P_M5"] = self.magnitude.probability_above(5)

        self.report["P_M55"] = self.magnitude.probability_above(5.5)

        self.report["P_M6"] = self.magnitude.probability_above(6.0)

        ############################################################
        # Áreas
        ############################################################

        self.report["TemporalArea"] = np.trapz(

            self.temporal.temporal_density,

            self.temporal.time_grid

        )

        self.report["MagnitudeArea"] = np.trapz(

            self.magnitude.magnitude_density,

            self.magnitude.magnitude_grid

        )

        ############################################################

        self.report_ready = True

        return self.report
        
    
        ##############################################################################
        # Imprimir reporte final
        ##############################################################################

    def print_report(self):

        """
        Imprime el reporte final del pronóstico.
        """

        if not self.report_ready:

            self.build_report()

        print()

        print("="*62)
        print("         NEURAL POINT PROCESS FORECAST REPORT")
        print("="*62)

        ############################################################
        # Último evento
        ############################################################

        print()
        print("Último evento observado")
        print("-"*62)

        print("Tiempo               : {:.6f} horas".format(
            self.report["last_time"]))

        print("Magnitud             : {:.5f}".format(
            self.report["last_magnitude"]))

        ############################################################
        # Pronóstico temporal
        ############################################################

        print()
        print("="*62)
        print("PRONÓSTICO TEMPORAL")
        print("="*62)

        print()

        print("Probabilidad ≤ 1 día : {:6.2f} %".format(
            self.report["P_1_day"]*100))

        print("Probabilidad ≤ 7 días: {:6.2f} %".format(
            self.report["P_7_days"]*100))

        print("Probabilidad ≤30 días: {:6.2f} %".format(
            self.report["P_30_days"]*100))

        print()

        print("Área densidad        : {:.6f}".format(
            self.report["TemporalArea"]))

        ############################################################
        # Pronóstico magnitud
        ############################################################

        print()
        print("="*62)
        print("PRONÓSTICO DE MAGNITUD")
        print("="*62)

        print()

        print("P(M ≥ 3.0)           : {:6.2f} %".format(
            self.report["P_M3"]*100))

        print("P(M ≥ 5)             : {:6.2f} %".format(
            self.report["P_M5"]*100))

        print("P(M ≥ 5.5)           : {:6.3f} %".format(
            self.report["P_M55"]*100))

        print("P(M ≥ 6.0)           : {:6.3f} %".format(
            self.report["P_M6"]*100))

        print()

        print("Área densidad        : {:.6f}".format(
            self.report["MagnitudeArea"]))

        ############################################################
        # Estado
        ############################################################

        print()
        print("="*62)
        print("ESTADO")
        print("="*62)

        print()

        print("Forecast Temporal    : OK")

        print("Forecast Magnitud    : OK")

        print("Reporte              : OK")

        print()

        print("="*62)