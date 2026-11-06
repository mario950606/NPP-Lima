"""Perfil de magnitud para el NPP-LSTM Softplus W20.

No reentrena. Calcula F(m) de la rama de magnitud en una malla,
la diferencia de F en un rango finito y una distribución relativa
normalizada SOLO a ese rango. No es una predicción exacta de magnitud,
ni una probabilidad de que ocurra un sismo dentro de un horizonte temporal.
"""

import numpy as np


def perfil_magnitud(modelo, hist_t, hist_m, horas_desde_ultimo,
                    minimo=4.5, maximo=8.5, puntos=401):
    """Devuelve curva de densidad condicional al rango mostrado o falla explícitamente.

    Args:
      modelo: modelo Keras con entradas [hist_t, hist_m, dt, mag], salidas [H, F].
      hist_t, hist_m: arreglos de dimensiones (1, ventana, 1).
      horas_desde_ultimo: tiempo transcurrido hasta la fecha de consulta.
      minimo, maximo: rango de magnitudes mostrado.
      puntos: número de puntos en la malla.

    La normalización se hace con F(maximo)-F(minimo). Esta masa no tiene
    por qué ser uno. Los porcentajes de intervalos son condicionales a
    que el evento objetivo esté dentro del rango [minimo, maximo].
    """
    if not (np.isfinite(minimo) and np.isfinite(maximo) and maximo > minimo):
        raise ValueError("Rango de magnitud inválido.")
    if puntos < 51:
        raise ValueError("Usa al menos 51 puntos de magnitud.")
    if not np.isfinite(horas_desde_ultimo) or horas_desde_ultimo < 0:
        raise ValueError("Tiempo desde el último sismo inválido.")

    hist_t = np.asarray(hist_t, dtype=np.float32)
    hist_m = np.asarray(hist_m, dtype=np.float32)
    if hist_t.ndim != 3 or hist_t.shape != hist_m.shape or hist_t.shape[0] != 1:
        raise ValueError("Los historiales deben tener forma (1, ventana, 1).")
    if np.any(hist_t <= 0) or np.any(hist_m <= 0):
        raise ValueError("Historiales inválidos: no se puede calcular su logaritmo.")

    malla = np.linspace(minimo, maximo, puntos, dtype=np.float32)
    x_t = np.repeat(hist_t, puntos, axis=0)
    x_m = np.repeat(hist_m, puntos, axis=0)
    dt = np.full((puntos, 1), max(float(horas_desde_ultimo), 1e-3), dtype=np.float32)
    m = malla.reshape(-1, 1)

    # Solo realizamos inferencia. Keras acepta arreglos NumPy como entrada.
    _, f_tensor = modelo([x_t, x_m, dt, m], training=False)
    f = np.asarray(f_tensor, dtype=np.float64).reshape(-1)
    if f.size != puntos or not np.isfinite(f).all():
        raise ValueError("La salida de magnitud F(m) contiene valores inválidos.")
    if np.any(np.diff(f) < -1e-4):
        raise ValueError("F(m) no es monótona: no puede interpretarse como CDF.")

    # Tolerancia para redondeos numéricos diminutos.
    f = np.maximum.accumulate(f)
    masa_bruta = float(f[-1] - f[0])
    if masa_bruta <= 1e-6:
        raise ValueError(
            "El modelo asigna masa prácticamente nula a [4.5, 8.5]; "
            "no es válido mostrar porcentajes normalizados."
        )

    cdf = np.clip((f - f[0]) / masa_bruta, 0.0, 1.0)
    densidad = np.maximum(np.gradient(cdf, malla.astype(np.float64)), 0.0)
    # Corregir el pequeño error de integración numérica de la derivada.
    integral = float(np.sum((densidad[1:] + densidad[:-1]) / 2 * np.diff(malla)))
    if not np.isfinite(integral) or integral <= 0:
        raise ValueError("No se puede normalizar la densidad de magnitud.")
    densidad /= integral

    def prob_intervalo(desde, hasta):
        if not minimo <= desde < hasta <= maximo:
            raise ValueError("Intervalo fuera de la malla.")
        return float(np.interp(hasta, malla, cdf) - np.interp(desde, malla, cdf))

    intervalos = [
        (4.5, 5.0, "Mw 4.5 a <5.0"),
        (5.0, 6.0, "Mw 5.0 a <6.0"),
        (6.0, 8.5, "Mw 6.0 a 8.5"),
    ]
    # A propósito se usan exactamente los límites 4.5 y 8.5 del experimento.
    if not (abs(minimo - 4.5) < 1e-7 and abs(maximo - 8.5) < 1e-7):
        intervalos = [(minimo, maximo, f"Mw {minimo:.1f} a {maximo:.1f}")]

    return {
        "magnitudes": malla,
        "cdf_relativa": cdf,
        "densidad_relativa": densidad,
        "masa_original_en_rango": masa_bruta,
        "mediana_condicional": float(np.interp(0.5, cdf, malla)),
        "moda_aproximada": float(malla[np.argmax(densidad)]),
        "intervalos": [(etiqueta, prob_intervalo(a, b)) for a, b, etiqueta in intervalos],
    }
