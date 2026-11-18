"""Detección de intervenciones del operario en el histórico de ocupación.

La red ya tiene una furgoneta repartiendo bicicletas, así que los cambios que
registra el scraper mezclan dos cosas: movimientos de clientes e intervenciones
del operario. Aquí separamos las que podemos atribuir a una intervención con
mucha confianza, para no calcular la demanda ni el contrafactual con datos
"ya corregidos" por la operación real.

La regla es deliberadamente conservadora. Preferimos dejar pasar alguna
intervención antes que borrar demanda real, así que solo marcamos:

1. Saltos de 3 o más bicicletas entre lecturas separadas por 10 minutos o menos
   en horario de bajo uso (23:00 a 07:00), cuando no hay clientes suficientes
   para explicarlos.
2. Saltos de 5 o más bicicletas a cualquier hora.

El resto de saltos grandes se guardan como "probables" solo para el informe,
nunca para corregir los datos.

La serie contrafactual se ancla a cada hora natural, no al inicio del histórico:
así el recorte en 0 no se acumula durante meses y no convertimos una red con
demanda desequilibrada en una red vacía.
"""

import numpy as np
import pandas as pd

UMBRAL_SALTO = 3
UMBRAL_SALTO_GRANDE = 5
DT_MAX_MIN = 10
HORAS_BAJO_USO = (23, 0, 1, 2, 3, 4, 5, 6)


def marcar_intervenciones(df, columna='bicis_disponibles'):
    """Añadimos delta, intervalo y una marca de intervención a cada lectura.

    Requiere las columnas id_estacion y timestamp. Devuelve el DataFrame
    ordenado por estación y tiempo con las columnas nuevas.
    """
    trabajo = df[['id_estacion', 'timestamp', columna]].copy()
    trabajo = trabajo.sort_values(['id_estacion', 'timestamp']).reset_index(drop=True)

    agrupado = trabajo.groupby('id_estacion')
    trabajo['delta'] = agrupado[columna].diff()
    trabajo['dt_min'] = agrupado['timestamp'].diff().dt.total_seconds() / 60

    # Solo consideramos intervalos cortos: un salto repartido en media hora no
    # es un traslado de furgoneta.
    intervalo_valido = trabajo['dt_min'] <= DT_MAX_MIN
    magnitud = trabajo['delta'].abs()
    hora = trabajo['timestamp'].dt.hour
    bajo_uso = hora.isin(HORAS_BAJO_USO)

    trabajo['confianza'] = 'ninguna'
    trabajo.loc[magnitud >= UMBRAL_SALTO, 'confianza'] = 'probable'
    trabajo.loc[intervalo_valido & (((magnitud >= UMBRAL_SALTO) & bajo_uso)
                                    | (magnitud >= UMBRAL_SALTO_GRANDE)), 'confianza'] = 'alta'
    trabajo['es_intervencion'] = trabajo['confianza'] == 'alta'

    return trabajo


def serie_sin_intervenciones(df, columna='bicis_disponibles', columna_capacidad='capacidad'):
    """Reconstruimos la serie de bicis quitando las intervenciones detectadas.

    Aplicamos solo los cambios que quedan tras anular los de alta confianza, pero
    anclando cada hora natural (y cada hueco de más de 10 minutos) al valor
    observado. Sin ese ancla, el recorte en 0 se acumularía durante meses y el
    contrafactual acabaría con la red entera vacía, que no es un escenario
    creíble sino un artefacto aritmético.

    El resultado es un contrafactual local: responde a "qué habría pasado en esta
    hora si el operario no hubiera intervenido". No reconstruye cadenas de varias
    horas, ni la demanda que se perdió por no haber bicis, así que sigue siendo
    una cota inferior del daño real.
    """
    marca = marcar_intervenciones(df, columna=columna)
    marca['delta_cliente'] = marca['delta'].where(~marca['es_intervencion'], 0.0).fillna(0.0)
    marca = marca.sort_values(['id_estacion', 'timestamp']).reset_index(drop=True)

    if columna_capacidad in df.columns and columna_capacidad not in marca.columns:
        marca = marca.merge(
            df[['id_estacion', 'timestamp', columna_capacidad]],
            on=['id_estacion', 'timestamp'], how='left'
        )
    capacidad = (marca[columna_capacidad] if columna_capacidad in marca.columns
                 else pd.Series(np.inf, index=marca.index))

    # Bloques: cada hora natural y cada hueco largo empiezan de cero.
    hora = marca['timestamp'].dt.hour
    cambio_hora = hora != hora.groupby(marca['id_estacion']).shift()
    hueco = marca['dt_min'] > DT_MAX_MIN
    marca['bloque'] = (cambio_hora | hueco).groupby(marca['id_estacion']).cumsum()

    grupos = marca.groupby(['id_estacion', 'bloque'])
    marca['ancla'] = grupos[columna].transform('first')
    marca['_acumulado'] = grupos['delta_cliente'].cumsum()
    marca['bicis_sin_van'] = (marca['ancla'] + marca['_acumulado']).clip(0, capacidad)

    return marca


def resumen_intervenciones(marca):
    """Resumimos cuánto de la actividad observada parece del operario.

    Devolvemos dos lecturas: la confirmada (la que usamos para corregir) y el
    techo de lo probable. Cualquier cifra intermedia es defendible, así que los
    informes deben citar el rango, no un valor único.
    """
    actividad = float(marca['delta'].abs().sum())
    magnitud = marca['delta'].abs()
    intervalo_valido = marca['dt_min'] <= DT_MAX_MIN

    alta = marca['confianza'] == 'alta'
    probable = intervalo_valido & (magnitud >= UMBRAL_SALTO)
    dias = marca['timestamp'].dt.date.nunique()

    return {
        'actividad_total_bicis': int(actividad),
        'eventos_alta_confianza': int(alta.sum()),
        'bicis_alta_confianza': int(magnitud[alta].sum()),
        'pct_alta_confianza': round(magnitud[alta].sum() / max(actividad, 1) * 100, 1),
        'bicis_por_dia_alta_confianza': round(magnitud[alta].sum() / max(dias, 1), 1),
        'eventos_probables': int(probable.sum()),
        'bicis_probables': int(magnitud[probable].sum()),
        'pct_probables': round(magnitud[probable].sum() / max(actividad, 1) * 100, 1),
        'dias_analizados': int(dias)
    }
