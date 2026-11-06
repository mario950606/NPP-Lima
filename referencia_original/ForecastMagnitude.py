# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:47:10 2026

@author: Usuario
"""


# -*- coding: utf-8 -*-
"""
ForecastMagnitude.py

Motor de pronóstico probabilístico de magnitudes
utilizando un modelo Neural Point Process previamente
entrenado.

Este módulo NO reentrena el modelo.
"""

import numpy as np


class ForecastMagnitude(object):

    """
    Forecast probabilístico de magnitudes.

    El modelo debe haber sido reconstruido previamente por
    ForecastEngine.
    """

    #####################################################################
    # Constructor
    #####################################################################

    def __init__(
            self,
            state,
            model,
            min_magnitude=None,
            max_magnitude=8.5,
            n_points=500):

        self.state = state

        self.model = model

        ############################################################

        if min_magnitude is None:

            
            self.min_magnitude = float(state.M0pred)

        else:

            self.min_magnitude = float(min_magnitude)

        self.max_magnitude = float(max_magnitude)

        self.n_points = int(n_points)

        ############################################################

        self._validate_inputs()

        self._initialize_results()

    #####################################################################
    # Validación
    #####################################################################

    def _validate_inputs(self):

        if self.state is None:

            raise RuntimeError(
                "ForecastState no disponible."
            )

        if self.model is None:

            raise RuntimeError(
                "Modelo NeuralPP no disponible."
            )

        if not self.state.ready:

            raise RuntimeError(
                "ForecastEngine no ha sido inicializado."
            )

        if self.max_magnitude <= self.min_magnitude:

            raise ValueError(
                "max_magnitude debe ser mayor que min_magnitude."
            )

        if self.n_points <= 10:

            raise ValueError(
                "n_points debe ser mayor que 10."
            )

    #####################################################################
    # Inicialización
    #####################################################################

    def _initialize_results(self):

        self.magnitude_grid = None

        self.magnitude_density = None

        self.magnitude_cdf = None
        
        
        #####################################################################
        # Resumen
        #####################################################################

    def summary(self):

        print()

        print("=" * 70)
        print("FORECAST MAGNITUDE")
        print("=" * 70)

        print()

        print("Modelo disponible :", self.model is not None)

        print()

        print("Último evento")
        print("------------------------------")

        print("Tiempo     :", self.state.last_time)
        print("Magnitud   :", self.state.last_magnitude)

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

        print("Magnitude Grid :", self.magnitude_grid is not None)
        print("Density        :", self.magnitude_density is not None)
        print("CDF            :", self.magnitude_cdf is not None)

        print()

        print("=" * 70)

#####################################################################
# Construcción de la malla de magnitudes
#####################################################################

    def build_magnitude_grid(self):

        self.magnitude_grid = np.linspace(

            self.min_magnitude,

            self.max_magnitude,

            self.n_points

        )

        return self.magnitude_grid



##############################################################################
# Predicción del modelo
##############################################################################

    def _predict_model(self, magnitude):

        """
        Ejecuta una inferencia del modelo NeuralPP
        para una magnitud candidata.
        """

        ############################################################
        # Copias del estado
        ############################################################

        rnn_times = self.state.mag_RNN_times.copy()

        rnn_mags = self.state.mag_RNN_mags.copy()

        chfn = self.state.mag_CHFN.copy()
        

        cmfn = self.state.mag_CMFN.copy()

        mask = self.state.mag_mask.copy()
        EPS = 1.0 / 86400.0
        chfn[-1] = EPS

        ############################################################
        # Magnitud candidata
        ############################################################

        cmfn[-1] = magnitude

        ############################################################
        # Ejecutar modelo
        ############################################################

        outputs = self.model.model.predict(

            [

            rnn_times,

            rnn_mags,

            chfn,

            cmfn,

            mask

            ],

            batch_size=512,

            verbose=0

        )

        ############################################################

        hazard = outputs[2]

        cumulative = outputs[3]

        return hazard, cumulative



    ##############################################################################
    # Densidad a partir de las salidas del modelo
    ##############################################################################

    def _density_from_outputs(
            self,
            hazard,
            cumulative):

        """
        Calcula la densidad de magnitud.

        f(m)=λ(m)exp(-Λ(m))
        """

        density = hazard * np.exp(-cumulative)

        return float(density[-1])

    ##############################################################################
    # Calcular densidad
    ##############################################################################

    def compute_density(self):

        print()

        print("="*70)
        print("Calculando densidad de magnitudes...")
        print("="*70)

        if self.magnitude_grid is None:

            self.build_magnitude_grid()

        ############################################################

        density = np.zeros(len(self.magnitude_grid))

        ############################################################

        for i, magnitude in enumerate(self.magnitude_grid):

            hazard, cumulative = self._predict_model(magnitude)
            
            if np.isnan(hazard).any():

                raise RuntimeError(
                    "NaN detectado en hazard."
                    )

            if np.isnan(cumulative).any():

                raise RuntimeError(
                    "NaN detectado en cumulative."
                    )
            
            
            
            

            density[i] = self._density_from_outputs(

                hazard,

                cumulative

            )

            if (i+1) % 50 == 0:

                print("{:4d}/{:4d}".format(

                    i+1,

                    len(self.magnitude_grid)

                ))

        ############################################################

        self.magnitude_density = density

        print()

        print("Densidad calculada correctamente.")

        print()

        return density


    ##############################################################################
    # Calcular distribución acumulada
    ##############################################################################

    def compute_cumulative(self):

        """
        Calcula la distribución acumulada (CDF)
        mediante integración numérica usando
        la regla del trapecio.
        """

        if self.magnitude_density is None:

            self.compute_density()

        ############################################################

        cumulative = np.zeros(len(self.magnitude_density))

        ############################################################

        for i in range(1, len(self.magnitude_density)):

            dm = (

                self.magnitude_grid[i]

                -

                self.magnitude_grid[i-1]

            )

            cumulative[i] = (

                cumulative[i-1]

                +

                0.5 * (

                    self.magnitude_density[i]

                    +

                    self.magnitude_density[i-1]

                ) * dm

            )

        ############################################################

        self.magnitude_cdf = cumulative

        print()

        print("="*70)
        print("CDF calculada correctamente.")
        print()

        print("Probabilidad total :")

        print(cumulative[-1])

        print()

        return cumulative
    
    ##############################################################################
    # Probabilidad de excedencia
    ##############################################################################

    def probability_above(self, magnitude):

        """
        Calcula la probabilidad de que el próximo evento
        tenga una magnitud mayor o igual que 'magnitude'.

        Parameters
        ----------
        magnitude : float

        Returns
        -------
        float
        """

        if self.magnitude_cdf is None:

            self.compute_cumulative()

        ############################################################

        if magnitude <= self.magnitude_grid[0]:

            return float(self.magnitude_cdf[-1])

        ############################################################

        if magnitude >= self.magnitude_grid[-1]:

            return 0.0

        ############################################################

        cdf = np.interp(

            magnitude,

            self.magnitude_grid,

            self.magnitude_cdf

        )

        ############################################################

        probability = self.magnitude_cdf[-1] - cdf

        return float(probability)



