
import tensorflow as tf
from tensorflow.keras import Model, constraints, initializers, layers


class GlorotPositivo(initializers.Initializer):
    """Reproduce abs(glorot_uniform) del modelo original."""
    def __call__(self, shape, dtype=None):
        return tf.abs(initializers.GlorotUniform()(shape, dtype=dtype))


def construir_npp(ventana, mu_t, sigma_t, mu_m, sigma_m, m0pred=4.5):
    unidades = 64
    positivo = GlorotPositivo()
    no_negativo = constraints.NonNeg()

    # 1. Entradas
    hist_t = layers.Input((ventana, 1), name="hist_t")
    hist_m = layers.Input((ventana, 1), name="hist_m")
    dt = layers.Input((1,), name="tiempo_actual")
    mag = layers.Input((1,), name="magnitud_actual")

    # 2. Normalización original
    t_norm = layers.Lambda(
        lambda x: (tf.math.log(x) - mu_t) / sigma_t
    )(hist_t)

    m_norm = layers.Lambda(
        lambda x: (tf.math.log(x) - mu_m) / sigma_m
    )(hist_m)

    dt_norm = layers.Lambda(
        lambda x: (tf.math.log(x) - mu_t) / sigma_t
    )(dt)

    mag_shift = layers.Lambda(lambda x: x - m0pred)(mag)

    # 3. LSTM compartida
    historial = layers.Concatenate(axis=2)([t_norm, m_norm])
    memoria = layers.LSTM(
        unidades, activation="tanh", name="lstm"
    )(historial)

    # 4. Rama temporal
    aux_t = layers.Dense(
        unidades,
        use_bias=False,
        kernel_initializer=positivo,
        kernel_constraint=no_negativo
    )(dt_norm)

    rnn_t = layers.Dense(unidades)(memoria)

    h = layers.Activation("tanh")(
        layers.Add()([aux_t, rnn_t])
    )

    for _ in range(2):
        h = layers.Dense(
            unidades,
            activation="tanh",
            kernel_initializer=positivo,
            kernel_constraint=no_negativo
        )(h)

    H = layers.Dense(
        1,
        activation="softplus",
        kernel_initializer=positivo,
        kernel_constraint=no_negativo,
        name="riesgo_acumulado"
    )(h)

    # 5. Rama de magnitud: ReLU original
    aux_m = layers.Dense(
        unidades,
        activation="relu",
        use_bias=False,
        trainable=False,
        kernel_initializer=positivo,
        kernel_constraint=no_negativo
    )(mag_shift)

    rnn_m = layers.Dense(unidades)(memoria)

    z = layers.Activation("tanh")(
        layers.Add()([aux_m, rnn_m, aux_t])
    )

    # Capas ocultas de la rama de magnitud
    for _ in range(2):
        z = layers.Dense(
            unidades,
            activation="tanh",
            kernel_initializer=positivo,
            kernel_constraint=no_negativo
        )(z)    

    F = layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer=positivo,
        kernel_constraint=no_negativo,
        name="acumulada_magnitud"
    )(z)

    return Model(
        inputs=[hist_t, hist_m, dt, mag],
        outputs=[H, F],
        name="NPP_LSTM_original"
    )
