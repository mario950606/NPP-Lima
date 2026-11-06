# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 10:19:40 2026

@author: Usuario
"""

import numpy as np


# =============================================================================
# ForecastTemporal
# =============================================================================

class ForecastTemporal(object):

    """
    -----------------------------------------------------------------------
    ForecastTemporal

    Módulo encargado del pronóstico temporal utilizando un modelo NeuralPP
    previamente reconstruido por ForecastEngine.

    Este módulo NO:

        • entrena la red
        • carga checkpoints
        • reconstruye la LSTM

    Únicamente reutiliza el estado interno generado por ForecastEngine.

    -----------------------------------------------------------------------
    """

    ###########################################################################
    # Constructor
    ###########################################################################

    def __init__(self, state, model):

        self.state = state

        self.model = model

        self._validate_inputs()

        self._initialize_results()

    ###########################################################################
    # Validaciones
    ###########################################################################

    def _validate_inputs(self):

        """
        Verifica que ForecastEngine haya entregado correctamente
        el estado interno y el modelo NeuralPP.
        """

        if self.state is None:

            raise ValueError(
                "ForecastState no fue proporcionado."
            )

        if self.model is None:

            raise ValueError(
                "Modelo NeuralPP no fue proporcionado."
            )

        required = [

            "reshape_RNN_times",

            "reshape_RNN_mags",

            "reshape_CHFN",

            "reshape_CMFN",

            "reshape_mask",

            "last_time",

            "last_magnitude"

        ]

        for attribute in required:

            if not hasattr(self.state, attribute):

                raise AttributeError(

                    "ForecastState no contiene '{}'".format(attribute)

                )

    ###########################################################################
    # Inicialización
    ###########################################################################

    def _initialize_results(self):

        """
        Inicializa todas las variables utilizadas durante el pronóstico.

        Ningún cálculo se realiza en esta etapa.
        """

        ############################################################
        # Estado reutilizado
        ############################################################

        self.last_time = self.state.last_time

        self.last_magnitude = self.state.last_magnitude

        ############################################################
        # Configuración temporal
        ############################################################

        self.max_days = None

        self.n_points = None

        self.time_grid = None

        ############################################################
        # Resultados
        ############################################################

        self.temporal_density = None

        self.temporal_cdf = None

        ############################################################
        # Estadísticos (Fase 3)
        ############################################################

        self.expected_time = None

        self.most_probable_time = None

        self.median_time = None

        self.percentiles = {}

        self.probabilities = {}

        ############################################################

        self.ready = True

    ###########################################################################
    # Estado
    ###########################################################################

    def is_ready(self):

        return self.ready

    ###########################################################################
    # Resumen
    ###########################################################################

    def summary(self):

        print()

        print("=" * 70)

        print("FORECAST TEMPORAL")

        print("=" * 70)

        print()

        print("Modelo disponible :", self.ready)

        print()

        print("Último evento")

        print("------------------------------")

        print("Tiempo     :", self.last_time)

        print("Magnitud   :", self.last_magnitude)

        print()

        print("Estado recibido")

        print("------------------------------")

        print("RNN Times :", self.state.reshape_RNN_times.shape)

        print("RNN Mags  :", self.state.reshape_RNN_mags.shape)

        print("CHFN      :", self.state.reshape_CHFN.shape)

        print("CMFN      :", self.state.reshape_CMFN.shape)

        print("Mask      :", self.state.reshape_mask.shape)

        print()

        print("Resultados")

        print("------------------------------")

        print("Time Grid :", self.time_grid is not None)

        print("Density   :", self.temporal_density is not None)

        print("CDF       :", self.temporal_cdf is not None)

        print()

        print("=" * 70)

##############################################################################
# Construcción del eje temporal
##############################################################################

    def build_time_grid(
            self,
            max_days=30.0,
            n_points=500):

        """
        Construye el eje temporal donde será evaluado
        el modelo NeuralPP.

        Parameters
        ----------
        max_days : float

            Horizonte temporal del pronóstico.

        n_points : int

            Número de puntos.

        """

        self.max_days = float(max_days)

        self.n_points = int(n_points)

        
        
        EPSILON = 1.0 / 86400.0      # 1 segundo expresado en días

        self.time_grid = np.linspace(
            EPSILON,
            self.max_days,
            self.n_points
            )
        
        

        return self.time_grid

##############################################################################
# Predicción del modelo
##############################################################################

    def _predict_model(self, tau):

        """
        Ejecuta una inferencia del modelo NeuralPP
        para un tiempo tau.

        Parameters
        ----------
        tau : float

        Returns
        -------
        tuple

        hazard

        cumulative_hazard
        """

    ############################################################
    # Copias del estado
    ############################################################

        elapsed = self.state.reshape_CHFN.copy()

        elapsed[-1] = tau

    ############################################################
    # Ejecutar NeuralPP
    ############################################################

        outputs = self.model.model.predict(

            [

            self.state.reshape_RNN_times,

            self.state.reshape_RNN_mags,

            elapsed,

            self.state.reshape_CMFN,

            self.state.reshape_mask

        ],

        # batch_size=self.state.reshape_mask.shape[0], verbose=0
        batch_size=512, verbose=0

        )

    ############################################################

        hazard = outputs[0]

        cumulative = outputs[1]

        return hazard, cumulative

##############################################################################
# Conversión de las salidas del modelo
##############################################################################

    def _density_from_outputs(
            self,
            hazard,
            cumulative):

        """
        Convierte las salidas del modelo NeuralPP
        en una densidad temporal.

        f(t)=λ(t)exp(-Λ(t))
        """

    ############################################################

        h = float(hazard[-1])

        H = float(cumulative[-1])

    ############################################################

        density = h * np.exp(-H)

        return density
    
    
    def compute_density(self):

        """
        Evalúa la densidad temporal del próximo evento
        sobre toda la malla temporal.

        Returns
        -------
        numpy.ndarray
        """

    ############################################################
    # Verificaciones
    ############################################################

        if self.time_grid is None:

            raise RuntimeError(

                "Debe ejecutar build_time_grid() antes.")

    ############################################################

        density = np.zeros(

            self.n_points,

            dtype=np.float64)

    ############################################################

        print()

        print("="*70)

        print("Calculando densidad temporal...")

        print("="*70)

        ############################################################

        for i, tau in enumerate(self.time_grid):

        ########################################################

            hazard, cumulative = self._predict_model(tau)

        ########################################################

            density[i] = self._density_from_outputs(hazard,cumulative)

        ########################################################

            if (i+1) % 50 == 0:

                print(

                    "{:4d}/{:4d}".format(

                        i+1,

                        self.n_points

                    )

                )

    ############################################################

        self.temporal_density = density

    ############################################################

        print()

        print("Densidad calculada correctamente.")

        print()

        return density
    
    ##############################################################################
    # Cálculo de la distribución acumulada
    ##############################################################################

    def compute_cumulative(self):

        """
        Calcula la función de distribución acumulada (CDF)
        del tiempo hasta el próximo evento.

        Returns
        -------
        numpy.ndarray
        """

        ############################################################

        if self.temporal_density is None:

            raise RuntimeError(

                "Debe ejecutar compute_density() primero."

            )

    ############################################################
    # Espaciado temporal
    ############################################################

        dt = self.time_grid[1] - self.time_grid[0]

    ############################################################
    # Integración acumulativa (regla rectangular)
    ############################################################

        # cumulative = np.cumsum(

        #     self.temporal_density

        # ) * dt
        
        
        
        
        cumulative = np.zeros_like(self.temporal_density)

        for i in range(1, len(self.temporal_density)):

            cumulative[i] = (

                cumulative[i-1]

                + 0.5

                * (

                    self.temporal_density[i-1]

                    + self.temporal_density[i]

            )

        * dt

        )
        
        
        
        

    ############################################################
    # Normalización
    ############################################################

        total_probability = cumulative[-1]

        if total_probability > 0:

            cumulative /= total_probability

    ############################################################

        cumulative[0] = max(0.0, cumulative[0])

        cumulative[-1] = 1.0

    ############################################################

        self.temporal_cdf = cumulative

    ############################################################

        print()

        print("=" * 70)

        print("CDF calculada correctamente.")

        print()

        print("Probabilidad total :", total_probability)

        print()

        return cumulative
    
    ##############################################################################
    # Probabilidad antes de un tiempo dado
    ##############################################################################

    def probability_before(self, days):

        """
        Devuelve

        P(T<=days)

        Parameters
        ----------
        days : float
        """

        if self.temporal_cdf is None:

            raise RuntimeError(

                "Debe ejecutar compute_cumulative()."

            )

        probability = np.interp(

            days,

            self.time_grid,

            self.temporal_cdf

        )

        return float(probability)