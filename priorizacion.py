"""Reglas compartidas de priorización del motor de redistribución.

Sustituimos la urgencia binaria (100/10) por una puntuación continua que
combina criticidad, uso histórico de la estación y déficit previsto, y que
descuenta el tiempo de conducción. Así, entre dos estaciones igual de críticas
gana la que más demanda genera, sin ignorar una estación cercana por otra
lejana.
"""

# Pesos de la puntuación continua.
PESO_DEMANDA = 30.0
PESO_DEFICIT = 20.0
PESO_TIEMPO = 40.0

# Factor de seguridad para dimensionar el relleno de una estación.
FACTOR_SEGURIDAD_META = 1.5
RELLENO_MINIMO = 5


def meta_relleno(capacidad, salida_neta_hora):
    """Objetivo de bicis al llegar: escala con la salida neta esperada de la hora.

    Mantenemos el suelo histórico de 5 bicis y el techo de capacidad menos 2.
    """
    capacidad = int(capacidad)
    esperado = max(float(salida_neta_hora), 0.0) * FACTOR_SEGURIDAD_META
    return int(min(max(round(esperado), RELLENO_MINIMO), max(capacidad - 2, RELLENO_MINIMO)))


def score_destino(criticidad, demanda_norm, deficit):
    """Puntuación base de un destino, sin penalización por tiempo."""
    return (float(criticidad)
            + PESO_DEMANDA * float(demanda_norm)
            + PESO_DEFICIT * float(deficit))


def score_con_tiempo(score_base, tiempo_min, tiempo_max):
    """Restamos el coste de desplazamiento normalizado por el tiempo máximo de la red."""
    return float(score_base) - PESO_TIEMPO * (float(tiempo_min) / max(float(tiempo_max), 1.0))
