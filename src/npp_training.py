
import numpy as np
import tensorflow as tf

from utils import collate_times


def preparar_ventanas(tiempos, magnitudes, ventana=20, m0pred=4.5):
    """Reproduce las entradas creadas en NeuralPP.set_train_data()."""

    dT = np.ediff1d(tiempos)
    dM = np.delete(magnitudes, 0)
    n = len(dT)

    if np.any(dT <= 0):
        raise ValueError("Hay intervalos temporales no positivos.")

    hist_t = np.array([
        dT[i:i + ventana] for i in range(n - ventana)
    ]).reshape(-1, ventana, 1)

    hist_m = np.array([
        dM[i:i + ventana] for i in range(n - ventana)
    ]).reshape(-1, ventana, 1)

    dt = dT[ventana:].reshape(-1, 1)
    mag = dM[ventana:].reshape(-1, 1)

    _, mascara = collate_times(dT, dM, m0pred)
    mascara = mascara[ventana:].reshape(-1, 1)

    return tuple(
        a.astype("float32")
        for a in (hist_t, hist_m, dt, mag, mascara)
    )


def perdida_npp(modelo, lote, entrenando=False):
    """Log-verosimilitud del NPP original usando GradientTape."""

    hist_t, hist_m, dt, mag, mascara = lote

    # Derivadas respecto al tiempo y la magnitud
    with tf.GradientTape(
        persistent=True,
        watch_accessed_variables=False
    ) as tape:
        tape.watch([dt, mag])

        H, F = modelo(
            [hist_t, hist_m, dt, mag],
            training=entrenando
        )

        suma_H = tf.reduce_sum(H)
        suma_F = tf.reduce_sum(F)

    intensidad = tape.gradient(suma_H, dt)
    densidad = tape.gradient(suma_F, mag)
    del tape

    if intensidad is None or densidad is None:
        raise RuntimeError("No se pudieron calcular las derivadas.")

    epsilon = 1e-10

    # Log-verosimilitud temporal
    termino_t = (
        tf.math.log(tf.maximum(intensidad, 0.0) + epsilon)
        * mascara - H
    )

    # Log-verosimilitud de magnitud
    termino_m = (
        tf.math.log(tf.maximum(densidad, 0.0) + epsilon)
        * mascara
    )

    perdida = -tf.reduce_sum(termino_t + termino_m)

    return perdida, intensidad, densidad
