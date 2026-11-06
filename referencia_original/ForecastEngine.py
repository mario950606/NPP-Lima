# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 20:21:06 2026

@author: Usuario
"""

# -*- coding: utf-8 -*-
"""
===============================================================================

ForecastEngine.py

Neural Point Process Forecast Engine

Proyecto:
Forecasting the 2016–2017 Central Apennines Earthquake Sequence
With a Neural Point Process
(Stockman et al., 2023)

-------------------------------------------------------------------------------

OBJETIVO

Este módulo NO realiza pronósticos.

Su única responsabilidad es preparar completamente el modelo
entrenado para que posteriormente pueda consultarse mediante
ForecastTemporal.py y ForecastMagnitude.py.

Flujo

Catálogo
    │
    ▼
Lectura del catálogo
    │
    ▼
Separación Train/Test
    │
    ▼
Carga del Checkpoint
    │
    ▼
Reconstrucción del estado oculto
    │
    ▼
ForecastState

-------------------------------------------------------------------------------

Compatible con

Python          3.7.16
TensorFlow      2.1.0
NumPy           1.19.5
Pandas          1.2.3

===============================================================================
"""

import os

import numpy as np
import pandas as pd

from NeuralPP import NPP

from utils import (
    datetime_to_hours,
    truncate_catalog_by_threshold,
    append_burn_in_to_test_set
)


# =============================================================================
# ForecastState
# =============================================================================

class ForecastState(object):
    """
    Estado reconstruido del modelo Neural Point Process.

    Esta clase únicamente almacena información.

    No realiza ningún cálculo.
    """

    def __init__(self):

        ############################################################
        # Catálogo
        ############################################################

        self.catalog = None

        self.T_train = None
        self.M_train = None

        self.T_test = None
        self.M_test = None

        ############################################################
        # Modelo
        ############################################################

        self.npp = None

        self.checkpoint = None

        ############################################################
        # Estado temporal
        ############################################################

        self.reshape_RNN_times = None
        self.reshape_RNN_mags = None

        self.reshape_CHFN = None
        self.reshape_CMFN = None

        self.reshape_mask = None

        ############################################################
        # Estado magnitudes
        ############################################################

        self.mag_RNN_times = None
        self.mag_RNN_mags = None

        self.mag_CHFN = None
        self.mag_CMFN = None

        self.mag_mask = None

        ############################################################
        # Información auxiliar
        ############################################################

        self.partition = None

        self.Mcut = None

        self.M0pred = None

        self.time_step = None

        self.last_time = None

        self.last_magnitude = None

        self.number_of_events = None

        ############################################################

        self.ready = False


# =============================================================================
# ForecastEngine
# =============================================================================

class ForecastEngine(object):

    """
    Motor de inferencia del Neural Point Process.

    Este módulo reconstruye completamente el estado del modelo
    entrenado.

    No calcula:

        • pronósticos
        • probabilidades
        • gráficos

    """

    ###########################################################################

    def __init__(self,
                 earthquake_partition,
                 Mcut,
                 M0pred=4.5,
                 time_step=20):

        self.partition = earthquake_partition

        self.Mcut = float(Mcut)

        self.M0pred = float(M0pred)

        self.time_step = int(time_step)

        self.catalog = None

        self.npp = None

        self.checkpoint = None

        self.state = ForecastState()

    ###########################################################################
    # Partición entrenamiento / prueba
    ###########################################################################

    def select_training_testing_partition(self):

        partitions = {

            "Visso":250000,

            "IGP":450000,

            "Campotosto":500000

        }

        if self.partition not in partitions:

            raise ValueError(
                "Partición desconocida: {}".format(self.partition)
            )

        return partitions[self.partition]

    ###########################################################################
    # Nombre del checkpoint
    ###########################################################################

    def checkpoint_name(self):

        return (
            "checkpoint"
            + str(self.Mcut)
            + str(self.partition)
            + str(int(self.M0pred))
            + str(self.time_step)
        )

    ###########################################################################
    # Lectura del catálogo
    ###########################################################################

    def load_catalog(
            self,
            catalog_path="../data/Catalogs/SISMOS_LIMA_FORMATO_MODELO_cat.csv"):

        print("="*70)
        print("Leyendo catálogo...")
        print("="*70)

        if not os.path.isfile(catalog_path):

            raise FileNotFoundError(
                "No existe el catálogo:\n{}".format(catalog_path)
            )

        ############################################################

        catalog = pd.read_csv(catalog_path)

        catalog = datetime_to_hours(catalog)

        catalog = catalog.dropna()

        catalog = truncate_catalog_by_threshold(
            catalog,
            self.Mcut
        )

        ############################################################

        times = np.asarray(catalog["time"])

        mags = np.asarray(catalog["mw"])

        ############################################################

        timeupto = self.select_training_testing_partition()

        T_train = times[times < timeupto]

        M_train = mags[times < timeupto]

        T_test = times[times >= timeupto]

        M_test = mags[times >= timeupto]

        ############################################################
        # Burn-In
        ############################################################

        T_test, M_test = append_burn_in_to_test_set(

            T_train,

            T_test,

            M_train,

            M_test,

            self.time_step

        )

        ############################################################
        # Guardar información
        ############################################################

        self.catalog = catalog

        self.state.catalog = catalog

        self.state.T_train = T_train
        self.state.M_train = M_train

        self.state.T_test = T_test
        self.state.M_test = M_test
        
        
        ############################################################
        # Historial para reconstrucción de inferencia
        ############################################################

        self.state.history = (
            T_test.copy(),
            M_test.copy()
            )
        
        
        
        

        self.state.partition = self.partition

        self.state.Mcut = self.Mcut

        self.state.M0pred = self.M0pred

        self.state.time_step = self.time_step

        ############################################################

        print("Eventos entrenamiento :", len(T_train))
        print("Eventos prueba        :", len(T_test))
        print("Mcut                  :", self.Mcut)

        print("="*70)

        return self

    ###########################################################################
    # Cargar Checkpoint
    ###########################################################################

    def load_checkpoint(self):
        """
        Reconstruye la arquitectura NeuralPP y recupera los pesos
        previamente entrenados.

        No realiza entrenamiento.

        Returns
        -------
        ForecastEngine
        """

        print("=" * 70)
        print("Cargando modelo entrenado...")
        print("=" * 70)

        ############################################################
        # Verificaciones
        ############################################################

        if self.state.T_train is None:

            raise RuntimeError(
                "Debe ejecutar load_catalog() antes de load_checkpoint()."
            )

        ############################################################
        # Nombre del checkpoint
        ############################################################

        checkpoint = self.checkpoint_name()

        checkpoint_index = os.path.join(
            "../data",
            "Checkpoints",
            checkpoint + ".index"
        )

        if not os.path.isfile(checkpoint_index):

            raise FileNotFoundError(
                "\nNo existe el checkpoint:\n{}\n".format(
                    checkpoint_index
                )
            )

        ############################################################
        # Construcción de la arquitectura
        ############################################################

        print("Reconstruyendo NeuralPP...")

        self.npp = NPP(

            time_step=self.time_step,

            size_rnn=64,

            size_nn=64,

            size_layer_chfn=3,

            size_layer_cmfn=3,

            M0pred=self.M0pred

        )

        ############################################################
        # Inicializar estadísticas del entrenamiento
        ############################################################

        self.npp.set_train_data(

            self.state.T_train,

            self.state.M_train

        )

        ############################################################
        # Reconstrucción del modelo
        ############################################################

        self.npp.set_model(

            lam=0,

            stateful=False

        )

        ############################################################
        # Recuperar pesos
        ############################################################

        print("Recuperando pesos...")

        self.npp.load_weights(checkpoint)

        ############################################################
        # Guardar referencias
        ############################################################

        self.checkpoint = checkpoint

        self.state.npp = self.npp

        self.state.checkpoint = checkpoint

        ############################################################

        print()

        print("Checkpoint cargado correctamente.")

        print("Nombre :", checkpoint)

        print("=" * 70)

        return self


    ###########################################################################
    # Reconstrucción del estado oculto
    ###########################################################################

    def build_state(self):
        """
        Reconstruye completamente el estado interno de la red.

        El resultado es un ForecastState listo para que
        ForecastTemporal.py y ForecastMagnitude.py puedan consultar
        el modelo entrenado.

        Returns
        -------
        ForecastEngine
        """

        print("=" * 70)
        print("Reconstruyendo estado interno...")
        print("=" * 70)

        ############################################################
        # Verificaciones
        ############################################################

        if self.npp is None:

            raise RuntimeError(
                "Debe ejecutar load_checkpoint() primero."
            )

        ############################################################
        # Historial observado
        ############################################################

        hist = (

            self.state.T_test,

            self.state.M_test

        )

        ############################################################
        # Estado temporal
        ############################################################

        (
            reshape_RNN_times,
            reshape_RNN_mags,
            reshape_CHFN,
            reshape_CMFN,
            reshape_mask

        ) = self.npp.timereshape(hist)

        ############################################################
        # Estado magnitudes
        ############################################################

        last_time = self.state.T_test[-1]

        (
            mag_RNN_times,
            mag_RNN_mags,
            mag_CHFN,
            mag_CMFN,
            mag_mask

        ) = self.npp.magreshape(

            hist,

            last_time

        )

        ############################################################
        # Guardar estado temporal
        ############################################################

        self.state.reshape_RNN_times = reshape_RNN_times

        self.state.reshape_RNN_mags = reshape_RNN_mags

        self.state.reshape_CHFN = reshape_CHFN

        self.state.reshape_CMFN = reshape_CMFN

        self.state.reshape_mask = reshape_mask

        ############################################################
        # Guardar estado magnitudes
        ############################################################

        self.state.mag_RNN_times = mag_RNN_times

        self.state.mag_RNN_mags = mag_RNN_mags

        self.state.mag_CHFN = mag_CHFN

        self.state.mag_CMFN = mag_CMFN

        self.state.mag_mask = mag_mask

        ############################################################
        # Información del último evento
        ############################################################

        self.state.last_time = float(

            self.state.T_test[-1]

        )

        self.state.last_magnitude = float(

            self.state.M_test[-1]

        )

        self.state.number_of_events = int(

            len(self.state.T_test)

        )

        ############################################################
        # Motor listo
        ############################################################

        self.state.ready = True

        ############################################################

        print()

        print("Estado reconstruido correctamente.")

        print()

        print("Último tiempo      :", self.state.last_time)

        print("Última magnitud    :", self.state.last_magnitude)

        print("Eventos utilizados :", self.state.number_of_events)

        print()

        print("Estado temporal")

        print("-------------------------------")

        print("RNN Times :", reshape_RNN_times.shape)

        print("RNN Mags  :", reshape_RNN_mags.shape)

        print("CHFN      :", reshape_CHFN.shape)

        print("CMFN      :", reshape_CMFN.shape)

        print()

        print("Estado magnitudes")

        print("-------------------------------")

        print("RNN Times :", mag_RNN_times.shape)

        print("RNN Mags  :", mag_RNN_mags.shape)

        print("CHFN      :", mag_CHFN.shape)

        print("CMFN      :", mag_CMFN.shape)

        print("=" * 70)

        return self

    ###########################################################################
    # Inicialización completa
    ###########################################################################

    def initialize(
            self,
            catalog_path="../data/Catalogs/SISMOS_LIMA_FORMATO_MODELO_cat.csv"):
        """
        Inicializa completamente el motor de inferencia.

        Flujo

            Leer catálogo
                    ↓
            Cargar modelo entrenado
                    ↓
            Reconstruir estado oculto

        Parameters
        ----------
        catalog_path : str

        Returns
        -------
        ForecastEngine
        """

        self.load_catalog(catalog_path)

        self.load_checkpoint()

        self.build_state()

        return self


    ###########################################################################
    # Obtener ForecastState
    ###########################################################################

    def get_state(self):

        """
        Devuelve el ForecastState reconstruido.
        """

        if not self.state.ready:

            raise RuntimeError(
                "El motor todavía no ha sido inicializado."
            )

        return self.state


    ###########################################################################
    # Obtener NeuralPP
    ###########################################################################

    def get_model(self):

        """
        Devuelve el modelo NeuralPP reconstruido.
        """

        if self.npp is None:

            raise RuntimeError(
                "El modelo todavía no ha sido cargado."
            )

        return self.npp


    ###########################################################################
    # Estado del motor
    ###########################################################################

    def is_ready(self):

        """
        Indica si el motor está listo para inferencia.
        """

        return self.state.ready


    ###########################################################################
    # Información general
    ###########################################################################

    def summary(self):

        """
        Imprime un resumen completo del motor.
        """

        print()

        print("=" * 70)
        print("FORECAST ENGINE")
        print("=" * 70)

        print()

        print("Partition              :", self.partition)

        print("Checkpoint             :", self.checkpoint)

        print("Mcut                   :", self.Mcut)

        print("M0pred                 :", self.M0pred)

        print("Time Step              :", self.time_step)

        print()

        print("Training Events        :", len(self.state.T_train))

        print("Testing Events         :", len(self.state.T_test))

        print()

        print("Last Event Time        :", self.state.last_time)

        print("Last Event Magnitude   :", self.state.last_magnitude)

        print()

        print("Forecast Engine Ready  :", self.is_ready())

        print()

        print("Temporal State")

        print("------------------------------")

        print("RNN Times :", self.state.reshape_RNN_times.shape)

        print("RNN Mags  :", self.state.reshape_RNN_mags.shape)

        print("CHFN      :", self.state.reshape_CHFN.shape)

        print("CMFN      :", self.state.reshape_CMFN.shape)

        print()

        print("Magnitude State")

        print("------------------------------")

        print("RNN Times :", self.state.mag_RNN_times.shape)

        print("RNN Mags  :", self.state.mag_RNN_mags.shape)

        print("CHFN      :", self.state.mag_CHFN.shape)

        print("CMFN      :", self.state.mag_CMFN.shape)

        print()

        print("=" * 70)


###############################################################################
# TEST
###############################################################################

if __name__ == "__main__":

    print()

    print("=" * 70)
    print("Forecast Engine")
    print("=" * 70)

    engine = ForecastEngine(

        earthquake_partition="IGP",

        Mcut=3.5,

        M0pred=4.5,

        time_step=20

    )

    engine.initialize()

    engine.summary()

    state = engine.get_state()

    model = engine.get_model()

    print()

    print("ForecastState reconstruido correctamente.")

    print()

    print("Modelo NeuralPP disponible para inferencia.")

    print()

    print("Entregable 1 finalizado correctamente.")

    print("=" * 70)
