from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

WEATHER_CACHE_PATH = DATA_DIR / "nasa_power_santo_amaro_2020_2025.csv"
OUTPUT_PATH = DATA_DIR / "irrigation_dataset_3.csv"

CROP_CYCLE_DAYS = 72

# Santo Amaro - BA
LATITUDE_DEG = -12.55
ALTITUDE_M = 42.0

# Condição inicial do perfil de água no início de cada ciclo.
# 1.00 = armazenamento inicial igual à TAW.
#
# Isto NÃO significa "100% de umidade volumétrica".
# Significa que o armazenamento de água disponível no perfil
# começa próximo à capacidade de campo adotada no modelo.
INITIAL_STORAGE_FRACTION = 1.00

# Fração da TAW que pode ser depletada antes da irrigação.
#
# Para Coriandrum sativum:
# p = 0.30
#
# RAW = p * TAW
#
# Assim, a irrigação é acionada quando a depleção atinge
# 30% da água total disponível.
P_MAD = 0.30

# Limite de segurança do simulador.
# Não representa uma lâmina mínima nem uma recomendação
# agronômica fixa.
MAX_IRRIGATION_DEPTH_MM = 20.0


# ============================================================
# CENÁRIOS DE SOLO
#
# theta_fc = umidade volumétrica na capacidade de campo
# theta_wp = umidade volumétrica no ponto de murcha permanente
#
# Os cenários são sintéticos e representam diferentes
# condições de retenção de água do solo.
# ============================================================

SCENARIOS = [
    {"scenario_id": 1,  "theta_fc": 0.20, "theta_wp": 0.08},
    {"scenario_id": 2,  "theta_fc": 0.22, "theta_wp": 0.10},
    {"scenario_id": 3,  "theta_fc": 0.25, "theta_wp": 0.12},
    {"scenario_id": 4,  "theta_fc": 0.18, "theta_wp": 0.07},
    {"scenario_id": 5,  "theta_fc": 0.21, "theta_wp": 0.09},
    {"scenario_id": 6,  "theta_fc": 0.24, "theta_wp": 0.11},
    {"scenario_id": 7,  "theta_fc": 0.16, "theta_wp": 0.06},
    {"scenario_id": 8,  "theta_fc": 0.23, "theta_wp": 0.10},
    {"scenario_id": 9,  "theta_fc": 0.26, "theta_wp": 0.13},
    {"scenario_id": 10, "theta_fc": 0.19, "theta_wp": 0.08},
    {"scenario_id": 11, "theta_fc": 0.22, "theta_wp": 0.09},
    {"scenario_id": 12, "theta_fc": 0.27, "theta_wp": 0.14},
    {"scenario_id": 13, "theta_fc": 0.28, "theta_wp": 0.15},
    {"scenario_id": 14, "theta_fc": 0.30, "theta_wp": 0.16},
]


# ============================================================
# Kc DO COENTRO
#
# Baseado em Silva et al. (2013), para coentro:
#
# inicial       = 0.82
# desenvolvimento = 1.03
# meio          = 1.07
# final         = 0.93
# ============================================================

def get_crop_stage(crop_day):
    """
    Ciclo de 72 dias:

    1 -> inicial
    2 -> desenvolvimento
    3 -> meio
    4 -> final
    """

    if crop_day <= 15:
        return 1

    elif crop_day <= 30:
        return 2

    elif crop_day <= 55:
        return 3

    else:
        return 4


def get_kc(crop_stage):
    kc_values = {
        1: 0.82,
        2: 1.03,
        3: 1.07,
        4: 0.93,
    }

    return kc_values[crop_stage]


# ============================================================
# PROFUNDIDADE RADICULAR
#
# Representação sintética baseada na evolução observada
# no estudo brasileiro utilizado como referência.
#
# Início: ~2.75 cm
# Dia 15: ~6 cm
# Dia 30: ~10 cm
# Dia 55: ~13 cm
# Final: ~13 cm
# ============================================================

def get_root_depth(crop_day):

    if crop_day <= 15:

        fraction = (
            (crop_day - 1)
            / 14
        )

        return (
            0.0275
            + fraction
            * (
                0.0600
                - 0.0275
            )
        )

    elif crop_day <= 30:

        fraction = (
            (crop_day - 15)
            / 15
        )

        return (
            0.0600
            + fraction
            * (
                0.1000
                - 0.0600
            )
        )

    elif crop_day <= 55:

        fraction = (
            (crop_day - 30)
            / 25
        )

        return (
            0.1000
            + fraction
            * (
                0.1300
                - 0.1000
            )
        )

    else:

        return 0.1300


# ============================================================
# FAO-56
# ============================================================

def saturation_vapor_pressure(
    temperature_c,
):

    return (
        0.6108
        * np.exp(
            (
                17.27
                * temperature_c
            )
            / (
                temperature_c
                + 237.3
            )
        )
    )


def calculate_daily_eto(daily):

    """
    ETo diária pelo método FAO-56 Penman-Monteith.

    Entradas:

        Tmax
        Tmin
        Tmean
        RH média
        vento a 2 m
        radiação solar diária

    Saída:

        ETo em mm/dia
    """

    lat_rad = math.radians(
        LATITUDE_DEG
    )

    j = (
        daily["day_of_year"]
        .to_numpy(dtype=float)
    )

    tmax = (
        daily["temperature_max_c"]
        .to_numpy(dtype=float)
    )

    tmin = (
        daily["temperature_min_c"]
        .to_numpy(dtype=float)
    )

    tmean = (
        daily["temperature_mean_c"]
        .to_numpy(dtype=float)
    )

    rh = np.clip(
        daily[
            "air_humidity_mean_percent"
        ].to_numpy(dtype=float),
        0,
        100,
    )

    u2 = np.maximum(
        daily[
            "wind_speed_mean_m_s"
        ].to_numpy(dtype=float),
        0,
    )

    # Os dados da série climática utilizada já estão
    # representados em MJ/m²/dia.
    rs = np.maximum(
        daily[
            "solar_radiation_daily_mj_m2"
        ].to_numpy(dtype=float),
        0,
    )

    # --------------------------------------------------------
    # Pressão atmosférica
    # --------------------------------------------------------

    pressure_kpa = (
        101.3
        * (
            (
                293.0
                - 0.0065 * ALTITUDE_M
            )
            / 293.0
        )
        ** 5.26
    )

    gamma = (
        0.000665
        * pressure_kpa
    )

    # --------------------------------------------------------
    # Pressão de vapor
    # --------------------------------------------------------

    es_tmax = (
        saturation_vapor_pressure(
            tmax
        )
    )

    es_tmin = (
        saturation_vapor_pressure(
            tmin
        )
    )

    es = (
        es_tmax
        + es_tmin
    ) / 2.0

    ea = (
        es
        * (
            rh
            / 100.0
        )
    )

    vapor_pressure_deficit = np.maximum(
        es - ea,
        0,
    )

    # --------------------------------------------------------
    # Delta
    # --------------------------------------------------------

    delta = (
        4098.0
        * saturation_vapor_pressure(
            tmean
        )
        / (
            tmean
            + 237.3
        ) ** 2
    )

    # --------------------------------------------------------
    # Radiação extraterrestre
    # --------------------------------------------------------

    dr = (
        1.0
        + 0.033
        * np.cos(
            2.0
            * np.pi
            * j
            / 365.0
        )
    )

    solar_declination = (
        0.409
        * np.sin(
            2.0
            * np.pi
            * j
            / 365.0
            - 1.39
        )
    )

    sunset_angle = np.arccos(
        np.clip(
            -np.tan(lat_rad)
            * np.tan(
                solar_declination
            ),
            -1.0,
            1.0,
        )
    )

    gsc = 0.0820

    ra = (
        (
            24.0
            * 60.0
            / np.pi
        )
        * gsc
        * dr
        * (
            sunset_angle
            * np.sin(lat_rad)
            * np.sin(
                solar_declination
            )
            + np.cos(lat_rad)
            * np.cos(
                solar_declination
            )
            * np.sin(
                sunset_angle
            )
        )
    )

    # --------------------------------------------------------
    # Radiação de céu claro
    # --------------------------------------------------------

    rso = (
        0.75
        + 2e-5 * ALTITUDE_M
    ) * ra

    rso = np.maximum(
        rso,
        0.01,
    )

    # --------------------------------------------------------
    # Radiação líquida de ondas curtas
    # --------------------------------------------------------

    albedo = 0.23

    rns = (
        1.0 - albedo
    ) * rs

    # --------------------------------------------------------
    # Radiação líquida de ondas longas
    # --------------------------------------------------------

    tmax_k = (
        tmax
        + 273.16
    )

    tmin_k = (
        tmin
        + 273.16
    )

    ratio = np.divide(
        rs,
        rso,
        out=np.zeros_like(rs),
        where=rso > 0,
    )

    ratio = np.clip(
        ratio,
        0.3,
        1.0,
    )

    rnl = (
        4.903e-9
        * (
            (
                tmax_k ** 4
                + tmin_k ** 4
            )
            / 2.0
        )
        * (
            0.34
            - 0.14
            * np.sqrt(
                np.maximum(
                    ea,
                    0,
                )
            )
        )
        * (
            1.35
            * ratio
            - 0.35
        )
    )

    # --------------------------------------------------------
    # Radiação líquida
    # --------------------------------------------------------

    rn = (
        rns
        - rnl
    )

    g = 0.0

    # --------------------------------------------------------
    # FAO-56 Penman-Monteith
    # --------------------------------------------------------

    numerator = (
        0.408
        * delta
        * (
            rn
            - g
        )
        + gamma
        * (
            900.0
            / (
                tmean
                + 273.0
            )
        )
        * u2
        * vapor_pressure_deficit
    )

    denominator = (
        delta
        + gamma
        * (
            1.0
            + 0.34 * u2
        )
    )

    eto = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(
            numerator
        ),
        where=denominator > 0,
    )

    eto = np.maximum(
        eto,
        0.0,
    )

    result = daily[
        ["date"]
    ].copy()

    result[
        "et0_daily_mm"
    ] = eto

    return result


# ============================================================
# CARREGAMENTO DO CLIMA
# ============================================================

def load_weather():

    if not WEATHER_CACHE_PATH.exists():

        raise FileNotFoundError(
            f"Arquivo climático não encontrado:\n"
            f"{WEATHER_CACHE_PATH}"
        )

    weather = pd.read_csv(
        WEATHER_CACHE_PATH,
        parse_dates=[
            "timestamp"
        ],
    )

    required_columns = [
        "timestamp",
        "temperature_c",
        "air_humidity_percent",
        "wind_speed_m_s",
        "solar_radiation_kwh_m2",
        "precipitation_mm",
    ]

    missing = [
        column
        for column in required_columns
        if column not in weather.columns
    ]

    if missing:

        raise ValueError(
            "Colunas ausentes no arquivo climático: "
            + ", ".join(missing)
        )

    weather = weather[
        required_columns
    ].copy()

    weather = weather.dropna(
        subset=[
            "timestamp"
        ]
    )

    weather = weather.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    weather[
        "temperature_c"
    ] = pd.to_numeric(
        weather[
            "temperature_c"
        ],
        errors="coerce",
    )

    weather[
        "air_humidity_percent"
    ] = pd.to_numeric(
        weather[
            "air_humidity_percent"
        ],
        errors="coerce",
    ).clip(
        0,
        100,
    )

    weather[
        "wind_speed_m_s"
    ] = pd.to_numeric(
        weather[
            "wind_speed_m_s"
        ],
        errors="coerce",
    ).clip(
        lower=0,
    )

    weather[
        "solar_radiation_kwh_m2"
    ] = pd.to_numeric(
        weather[
            "solar_radiation_kwh_m2"
        ],
        errors="coerce",
    ).clip(
        lower=0,
    )

    weather[
        "precipitation_mm"
    ] = pd.to_numeric(
        weather[
            "precipitation_mm"
        ],
        errors="coerce",
    ).clip(
        lower=0,
    )

    weather = weather.dropna()

    return weather


# ============================================================
# ETO DIÁRIA
# ============================================================

def prepare_daily_eto(
    weather,
):

    weather = weather.copy()

    weather["date"] = (
        weather[
            "timestamp"
        ].dt.floor("D")
    )

    daily = (
        weather
        .groupby("date")
        .agg(
            temperature_max_c=(
                "temperature_c",
                "max",
            ),
            temperature_min_c=(
                "temperature_c",
                "min",
            ),
            temperature_mean_c=(
                "temperature_c",
                "mean",
            ),
            air_humidity_mean_percent=(
                "air_humidity_percent",
                "mean",
            ),
            wind_speed_mean_m_s=(
                "wind_speed_m_s",
                "mean",
            ),
            solar_radiation_daily_mj_m2=(
                "solar_radiation_kwh_m2",
                "sum",
            ),
            precipitation_daily_mm=(
                "precipitation_mm",
                "sum",
            ),
        )
        .reset_index()
    )

    daily[
        "day_of_year"
    ] = (
        daily[
            "date"
        ].dt.dayofyear
    )

    eto = calculate_daily_eto(
        daily
    )

    daily = daily.merge(
        eto,
        on="date",
        how="left",
    )

    return daily


# ============================================================
# DISTRIBUIÇÃO HORÁRIA DA ETo
# ============================================================

def distribute_daily_eto(
    weather,
    daily,
):

    weather = weather.copy()

    weather["date"] = (
        weather[
            "timestamp"
        ].dt.floor("D")
    )

    weather = weather.merge(
        daily[
            [
                "date",
                "et0_daily_mm",
            ]
        ],
        on="date",
        how="left",
    )

    solar = weather[
        "solar_radiation_kwh_m2"
    ].clip(
        lower=0
    )

    daily_solar = (
        weather
        .groupby("date")[
            "solar_radiation_kwh_m2"
        ]
        .transform("sum")
    )

    solar_component = np.divide(
        solar,
        daily_solar,
        out=np.zeros(
            len(weather)
        ),
        where=(
            daily_solar
            .to_numpy()
            > 0
        ),
    )

    uniform_component = (
        1.0 / 24.0
    )

    weights = (
        0.80
        * solar_component
        + 0.20
        * uniform_component
    )

    weight_sum = (
        pd.Series(weights)
        .groupby(
            weather["date"]
        )
        .transform("sum")
        .to_numpy()
    )

    weights = np.divide(
        weights,
        weight_sum,
        out=np.zeros_like(
            weights
        ),
        where=weight_sum > 0,
    )

    weather["et0"] = (
        weather[
            "et0_daily_mm"
        ]
        * weights
    )

    weather["etc"] = 0.0

    return weather


# ============================================================
# GERAÇÃO DE UM CENÁRIO
# ============================================================

def generate_scenario(
    weather,
    scenario,
):

    df = weather.copy()

    scenario_id = (
        scenario["scenario_id"]
    )

    theta_fc = (
        scenario["theta_fc"]
    )

    theta_wp = (
        scenario["theta_wp"]
    )

    p = P_MAD

    # --------------------------------------------------------
    # Dia transcorrido
    # --------------------------------------------------------

    first_timestamp = (
        df["timestamp"].iloc[0]
    )

    elapsed_days = (
        (
            df["timestamp"]
            - first_timestamp
        )
        / pd.Timedelta(
            days=1
        )
    )

    elapsed_days = (
        elapsed_days
        .to_numpy()
    )

    # Identificador do ciclo.
    # Usado apenas para impedir que estados e janelas
    # atravessem de uma cultura para outra.
    df["cycle_id"] = (
        np.floor(
            elapsed_days
            / CROP_CYCLE_DAYS
        ).astype(int)
        + 1
    )

    # Dia dentro do ciclo.
    df["crop_day"] = (
        np.floor(
            elapsed_days
        ).astype(int)
        % CROP_CYCLE_DAYS
    ) + 1

    # --------------------------------------------------------
    # Estágio
    # --------------------------------------------------------

    df["crop_stage"] = (
        df["crop_day"]
        .apply(
            get_crop_stage
        )
        .astype(float)
    )

    df["kc"] = (
        df["crop_stage"]
        .apply(get_kc)
        .astype(float)
    )

    # --------------------------------------------------------
    # Profundidade radicular
    # --------------------------------------------------------

    df["root_depth_m"] = (
        df["crop_day"]
        .apply(
            get_root_depth
        )
    )

    # --------------------------------------------------------
    # TAW / RAW
    # --------------------------------------------------------

    df["theta_fc"] = (
        theta_fc
    )

    df["theta_wp"] = (
        theta_wp
    )

    df["p"] = p

    df["taw_mm"] = (
        1000.0
        * (
            theta_fc
            - theta_wp
        )
        * df["root_depth_m"]
    )

    df["raw_mm"] = (
        p
        * df["taw_mm"]
    )

    # --------------------------------------------------------
    # ETc
    # --------------------------------------------------------

    df["etc"] = (
        df["et0"]
        * df["kc"]
    )

    # --------------------------------------------------------
    # BALANÇO HÍDRICO HORÁRIO
    # --------------------------------------------------------

    storage = (
        float(
            df["taw_mm"].iloc[0]
        )
        * INITIAL_STORAGE_FRACTION
    )

    previous_cycle = None

    storage_before_list = []
    storage_after_rain_list = []
    storage_after_et_list = []
    storage_after_irrigation_list = []

    soil_moisture_list = []
    depletion_list = []

    irrigation_event_list = []
    irrigation_depth_list = []

    for _, row in df.iterrows():

        cycle_id = int(
            row["cycle_id"]
        )

        taw = float(
            row["taw_mm"]
        )

        raw = float(
            row["raw_mm"]
        )

        # ----------------------------------------------------
        # Novo ciclo de cultivo
        # ----------------------------------------------------

        if (
            previous_cycle is None
            or cycle_id
            != previous_cycle
        ):

            storage = (
                taw
                * INITIAL_STORAGE_FRACTION
            )

        # ----------------------------------------------------
        # Se a TAW mudou devido ao crescimento radicular,
        # o armazenamento não pode ultrapassar a nova TAW.
        # ----------------------------------------------------

        storage = min(
            storage,
            taw,
        )

        storage_before = (
            storage
        )

        precipitation = max(
            0.0,
            float(
                row[
                    "precipitation_mm"
                ]
            ),
        )

        etc = max(
            0.0,
            float(
                row["etc"]
            ),
        )

        # ----------------------------------------------------
        # Chuva
        # ----------------------------------------------------

        storage_after_rain = min(
            taw,
            storage_before
            + precipitation,
        )

        # ----------------------------------------------------
        # Evapotranspiração
        # ----------------------------------------------------

        storage_after_et = max(
            0.0,
            storage_after_rain
            - etc,
        )

        # ----------------------------------------------------
        # Depleção
        # ----------------------------------------------------

        depletion = max(
            0.0,
            taw
            - storage_after_et,
        )

        # ----------------------------------------------------
        # Irrigação
        #
        # Aciona quando:
        #
        # depletion >= RAW
        #
        # A lâmina é o déficit necessário para retornar
        # ao armazenamento correspondente à TAW.
        # ----------------------------------------------------

        irrigation_event = (
            depletion >= raw
            and raw > 0
        )

        if irrigation_event:

            irrigation_depth = (
                taw
                - storage_after_et
            )

            irrigation_depth = max(
                0.0,
                irrigation_depth,
            )

            irrigation_depth = min(
                irrigation_depth,
                MAX_IRRIGATION_DEPTH_MM,
            )

        else:

            irrigation_depth = (
                0.0
            )

        storage_after_irrigation = min(
            taw,
            storage_after_et
            + irrigation_depth,
        )

        # ----------------------------------------------------
        # Estado observado ANTES da irrigação
        # ----------------------------------------------------

        if taw > 0:

            soil_moisture = (
                storage_after_et
                / taw
                * 100.0
            )

        else:

            soil_moisture = (
                0.0
            )

        # ----------------------------------------------------
        # Armazena estado
        # ----------------------------------------------------

        storage_before_list.append(
            storage_before
        )

        storage_after_rain_list.append(
            storage_after_rain
        )

        storage_after_et_list.append(
            storage_after_et
        )

        storage_after_irrigation_list.append(
            storage_after_irrigation
        )

        soil_moisture_list.append(
            soil_moisture
        )

        depletion_list.append(
            depletion
        )

        irrigation_event_list.append(
            bool(
                irrigation_event
            )
        )

        irrigation_depth_list.append(
            irrigation_depth
        )

        # ----------------------------------------------------
        # Próximo estado
        # ----------------------------------------------------

        storage = (
            storage_after_irrigation
        )

        previous_cycle = (
            cycle_id
        )

    # --------------------------------------------------------
    # Salva estados hídricos
    # --------------------------------------------------------

    df[
        "storage_before_mm"
    ] = storage_before_list

    df[
        "storage_after_rain_mm"
    ] = storage_after_rain_list

    df[
        "storage_after_et_mm"
    ] = storage_after_et_list

    df[
        "storage_after_irrigation_mm"
    ] = storage_after_irrigation_list

    df[
        "storage_mm"
    ] = (
        df[
            "storage_after_irrigation_mm"
        ]
    )

    df[
        "soil_moisture_percent"
    ] = soil_moisture_list

    df[
        "depletion_mm"
    ] = depletion_list

    df[
        "irrigation_event"
    ] = irrigation_event_list

    df[
        "irrigation_depth_mm"
    ] = irrigation_depth_list

    df[
        "scenario_id"
    ] = scenario_id

    # --------------------------------------------------------
    # LAGS
    #
    # Não atravessam ciclos.
    # --------------------------------------------------------

    df[
        "soil_moisture_lag_1h"
    ] = (
        df.groupby(
            "cycle_id"
        )[
            "soil_moisture_percent"
        ]
        .shift(1)
    )

    df[
        "soil_moisture_lag_3h"
    ] = (
        df.groupby(
            "cycle_id"
        )[
            "soil_moisture_percent"
        ]
        .shift(3)
    )

    df[
        "soil_moisture_lag_6h"
    ] = (
        df.groupby(
            "cycle_id"
        )[
            "soil_moisture_percent"
        ]
        .shift(6)
    )

    # --------------------------------------------------------
    # FEATURES DE JANELA
    #
    # Também não atravessam ciclos.
    # --------------------------------------------------------

    grouped = df.groupby(
        "cycle_id"
    )

    df[
        "etc_6h"
    ] = (
        grouped["etc"]
        .rolling(
            6,
            min_periods=1,
        )
        .sum()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    df[
        "etc_24h"
    ] = (
        grouped["etc"]
        .rolling(
            24,
            min_periods=1,
        )
        .sum()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    df[
        "rain_6h"
    ] = (
        grouped[
            "precipitation_mm"
        ]
        .rolling(
            6,
            min_periods=1,
        )
        .sum()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    df[
        "rain_24h"
    ] = (
        grouped[
            "precipitation_mm"
        ]
        .rolling(
            24,
            min_periods=1,
        )
        .sum()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    df[
        "temperature_6h_mean"
    ] = (
        grouped[
            "temperature_c"
        ]
        .rolling(
            6,
            min_periods=1,
        )
        .mean()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    df[
        "radiation_6h_sum"
    ] = (
        grouped[
            "solar_radiation_kwh_m2"
        ]
        .rolling(
            6,
            min_periods=1,
        )
        .sum()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    # --------------------------------------------------------
    # FEATURES TEMPORAIS
    # --------------------------------------------------------

    hour = (
        df["timestamp"].dt.hour
        + df["timestamp"].dt.minute
        / 60.0
    )

    day_of_year = (
        df[
            "timestamp"
        ].dt.dayofyear
    )

    df["hour_sin"] = np.sin(
        2.0
        * np.pi
        * hour
        / 24.0
    )

    df["hour_cos"] = np.cos(
        2.0
        * np.pi
        * hour
        / 24.0
    )

    df[
        "day_of_year_sin"
    ] = np.sin(
        2.0
        * np.pi
        * day_of_year
        / 365.0
    )

    df[
        "day_of_year_cos"
    ] = np.cos(
        2.0
        * np.pi
        * day_of_year
        / 365.0
    )

    # --------------------------------------------------------
    # Remove primeiras horas sem histórico suficiente.
    #
    # Como os lags são calculados por ciclo, as primeiras
    # 6 horas de cada ciclo são naturalmente removidas.
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "soil_moisture_lag_1h",
            "soil_moisture_lag_3h",
            "soil_moisture_lag_6h",
        ]
    ).reset_index(
        drop=True
    )

    # cycle_id é apenas controle interno.
    # Não será disponibilizado como feature.
    df = df.drop(
        columns=[
            "cycle_id"
        ]
    )

    return df


# ============================================================
# PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print(
        "GERADOR DATASET 3 - VERSÃO FINAL"
    )
    print("=" * 70)

    print(
        "\nParâmetros agronômicos:"
    )

    print(
        f"p / MAD: {P_MAD:.2f}"
    )

    print(
        f"Armazenamento inicial: "
        f"{INITIAL_STORAGE_FRACTION:.2f} × TAW"
    )

    print(
        f"Ciclo: "
        f"{CROP_CYCLE_DAYS} dias"
    )

    print(
        "\nCarregando dados climáticos..."
    )

    weather = load_weather()

    print(
        f"Registros climáticos: "
        f"{len(weather):,}"
    )

    print(
        f"Período: "
        f"{weather['timestamp'].min()} "
        f"→ "
        f"{weather['timestamp'].max()}"
    )

    print(
        "\nCalculando ETo diária FAO-56..."
    )

    daily = prepare_daily_eto(
        weather
    )

    print(
        f"ETo média diária: "
        f"{daily['et0_daily_mm'].mean():.3f} "
        f"mm/dia"
    )

    print(
        f"ETo mediana diária: "
        f"{daily['et0_daily_mm'].median():.3f} "
        f"mm/dia"
    )

    print(
        f"ETo P95: "
        f"{daily['et0_daily_mm'].quantile(.95):.3f} "
        f"mm/dia"
    )

    # --------------------------------------------------------
    # Distribuição horária
    # --------------------------------------------------------

    print(
        "\nDistribuindo ETo diária para as horas..."
    )

    weather = distribute_daily_eto(
        weather,
        daily,
    )

    # --------------------------------------------------------
    # Gera cenários
    # --------------------------------------------------------

    datasets = []

    for scenario in SCENARIOS:

        print(
            f"Gerando cenário "
            f"{scenario['scenario_id']:02d}/"
            f"{len(SCENARIOS)}..."
        )

        scenario_df = (
            generate_scenario(
                weather,
                scenario,
            )
        )

        datasets.append(
            scenario_df
        )

    # --------------------------------------------------------
    # Junta
    # --------------------------------------------------------

    dataset = pd.concat(
        datasets,
        ignore_index=True,
    )

    dataset = dataset.sort_values(
        [
            "scenario_id",
            "timestamp",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Organização das colunas
    # --------------------------------------------------------

    columns = [
        "timestamp",
        "scenario_id",

        "temperature_c",
        "air_humidity_percent",
        "wind_speed_m_s",
        "solar_radiation_kwh_m2",
        "precipitation_mm",

        "crop_day",
        "crop_stage",
        "root_depth_m",

        "theta_fc",
        "theta_wp",
        "p",

        "taw_mm",
        "raw_mm",

        "storage_before_mm",
        "storage_after_rain_mm",
        "storage_after_et_mm",
        "storage_after_irrigation_mm",
        "storage_mm",

        "soil_moisture_percent",
        "soil_moisture_lag_1h",
        "soil_moisture_lag_3h",
        "soil_moisture_lag_6h",

        "kc",
        "et0",
        "etc",
        "etc_6h",
        "etc_24h",

        "rain_6h",
        "rain_24h",

        "temperature_6h_mean",
        "radiation_6h_sum",

        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",

        "depletion_mm",

        "irrigation_event",
        "irrigation_depth_mm",
    ]

    dataset = dataset[
        columns
    ]

    # ========================================================
    # VALIDAÇÃO
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VALIDAÇÃO FINAL"
    )

    print(
        "=" * 70
    )

    print(
        f"Total de registros: "
        f"{len(dataset):,}"
    )

    print(
        f"Total de cenários: "
        f"{dataset['scenario_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Eventos
    # --------------------------------------------------------

    events = dataset[
        dataset[
            "irrigation_event"
        ]
    ]

    print(
        f"\nEventos de irrigação: "
        f"{len(events):,}"
    )

    print(
        f"Taxa de eventos: "
        f"{len(events) / len(dataset) * 100:.4f}%"
    )

    if len(events) > 0:

        print(
            "\nDistribuição das lâminas:"
        )

        print(
            f"Média:   "
            f"{events['irrigation_depth_mm'].mean():.3f} mm"
        )

        print(
            f"Mediana: "
            f"{events['irrigation_depth_mm'].median():.3f} mm"
        )

        print(
            f"P10:     "
            f"{events['irrigation_depth_mm'].quantile(.10):.3f} mm"
        )

        print(
            f"P25:     "
            f"{events['irrigation_depth_mm'].quantile(.25):.3f} mm"
        )

        print(
            f"P50:     "
            f"{events['irrigation_depth_mm'].quantile(.50):.3f} mm"
        )

        print(
            f"P75:     "
            f"{events['irrigation_depth_mm'].quantile(.75):.3f} mm"
        )

        print(
            f"P90:     "
            f"{events['irrigation_depth_mm'].quantile(.90):.3f} mm"
        )

        print(
            f"P95:     "
            f"{events['irrigation_depth_mm'].quantile(.95):.3f} mm"
        )

        print(
            f"P99:     "
            f"{events['irrigation_depth_mm'].quantile(.99):.3f} mm"
        )

        print(
            f"Máximo:  "
            f"{events['irrigation_depth_mm'].max():.3f} mm"
        )

        print(
            "\nEventos pequenos:"
        )

        print(
            f"< 0.50 mm: "
            f"{(events['irrigation_depth_mm'] < 0.50).sum():,}"
        )

        print(
            f"< 1.00 mm: "
            f"{(events['irrigation_depth_mm'] < 1.00).sum():,}"
        )

        print(
            f"< 2.00 mm: "
            f"{(events['irrigation_depth_mm'] < 2.00).sum():,}"
        )

        print(
            f"< 3.00 mm: "
            f"{(events['irrigation_depth_mm'] < 3.00).sum():,}"
        )

    # --------------------------------------------------------
    # ETc
    # --------------------------------------------------------

    dataset_copy = dataset.copy()

    dataset_copy[
        "date"
    ] = (
        dataset_copy[
            "timestamp"
        ].dt.floor("D")
    )

    daily_etc = (
        dataset_copy
        .groupby(
            [
                "scenario_id",
                "date",
            ]
        )[
            "etc"
        ]
        .sum()
    )

    print(
        f"\nETc média diária: "
        f"{daily_etc.mean():.3f} mm/dia"
    )

    print(
        f"ETc mediana diária: "
        f"{daily_etc.median():.3f} mm/dia"
    )

    if (
        daily_etc.mean()
        > 7.0
    ):

        print(
            "\nAVISO: ETc média acima de "
            "7 mm/dia. "
            "Verifique o arquivo climático."
        )

    # --------------------------------------------------------
    # TAW / RAW
    # --------------------------------------------------------

    print(
        f"\nTAW médio: "
        f"{dataset['taw_mm'].mean():.3f} mm"
    )

    print(
        f"TAW mínimo: "
        f"{dataset['taw_mm'].min():.3f} mm"
    )

    print(
        f"TAW máximo: "
        f"{dataset['taw_mm'].max():.3f} mm"
    )

    print(
        f"RAW médio: "
        f"{dataset['raw_mm'].mean():.3f} mm"
    )

    # --------------------------------------------------------
    # Umidade no acionamento
    # --------------------------------------------------------

    if len(events) > 0:

        print(
            "\nUmidade relativa do armazenamento "
            "no momento dos eventos:"
        )

        print(
            f"Média:   "
            f"{events['soil_moisture_percent'].mean():.3f}%"
        )

        print(
            f"Mediana: "
            f"{events['soil_moisture_percent'].median():.3f}%"
        )

        print(
            f"Mínimo:  "
            f"{events['soil_moisture_percent'].min():.3f}%"
        )

        print(
            f"Máximo:  "
            f"{events['soil_moisture_percent'].max():.3f}%"
        )

        print(
            "\nLimite teórico de acionamento:"
        )

        print(
            f"{(1.0 - P_MAD) * 100:.1f}% "
            "do armazenamento relativo"
        )

    # --------------------------------------------------------
    # Consistência do evento
    # --------------------------------------------------------

    event_condition_error = (
        dataset[
            "irrigation_event"
        ]
        != (
            dataset[
                "depletion_mm"
            ]
            >= dataset[
                "raw_mm"
            ]
        )
    ).sum()

    print(
        f"\nEventos inconsistentes com RAW: "
        f"{event_condition_error:,}"
    )

    # --------------------------------------------------------
    # Consistência do target
    # --------------------------------------------------------

    expected_depth = np.where(
        dataset[
            "irrigation_event"
        ],
        np.minimum(
            np.maximum(
                dataset[
                    "taw_mm"
                ]
                - dataset[
                    "storage_after_et_mm"
                ],
                0.0,
            ),
            MAX_IRRIGATION_DEPTH_MM,
        ),
        0.0,
    )

    target_error = np.max(
        np.abs(
            dataset[
                "irrigation_depth_mm"
            ]
            - expected_depth
        )
    )

    print(
        f"Erro máximo de consistência "
        f"do target: "
        f"{target_error:.12f} mm"
    )

    # --------------------------------------------------------
    # Estado após irrigação
    # --------------------------------------------------------

    storage_over_taw = (
        dataset[
            "storage_after_irrigation_mm"
        ]
        > dataset[
            "taw_mm"
        ]
        + 1e-9
    ).sum()

    print(
        f"Armazenamento acima da TAW: "
        f"{storage_over_taw:,}"
    )

    # --------------------------------------------------------
    # NaNs
    # --------------------------------------------------------

    print(
        f"NaNs: "
        f"{dataset.isna().sum().sum()}"
    )

    # --------------------------------------------------------
    # Lâminas negativas
    # --------------------------------------------------------

    print(
        f"Lâminas negativas: "
        f"{(
            dataset[
                'irrigation_depth_mm'
            ]
            < 0
        ).sum()}"
    )

    # --------------------------------------------------------
    # Lâminas acima do limite
    # --------------------------------------------------------

    print(
        f"Lâminas acima do limite: "
        f"{(
            dataset[
                'irrigation_depth_mm'
            ]
            > MAX_IRRIGATION_DEPTH_MM
        ).sum()}"
    )

    # --------------------------------------------------------
    # Umidade fora de 0–100%
    # --------------------------------------------------------

    moisture_invalid = (
        (
            dataset[
                "soil_moisture_percent"
            ]
            < 0
        )
        |
        (
            dataset[
                "soil_moisture_percent"
            ]
            > 100
        )
    ).sum()

    print(
        f"Umidade relativa fora de 0–100%: "
        f"{moisture_invalid}"
    )

    # --------------------------------------------------------
    # Salva
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DATASET 3 FINAL GERADO COM SUCESSO"
    )

    print(
        "=" * 70
    )

    print(
        f"\nArquivo:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"\nDimensões: "
        f"{dataset.shape[0]:,} linhas × "
        f"{dataset.shape[1]} colunas"
    )


if __name__ == "__main__":
    main()