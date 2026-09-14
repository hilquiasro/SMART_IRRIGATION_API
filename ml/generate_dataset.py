from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RANDOM_SEED = 42
CROP_CYCLE_DAYS = 50

STORAGE_CAPACITY_MM = 60.0
INITIAL_STORAGE_FRACTION = 0.70

# Fração efetiva da precipitação que realmente contribui
# para a água disponível no solo.
RAIN_EFFECTIVE_FRACTION = 0.75

# Margem de recuperação após atingir o limite inferior.
RECOVERY_MARGIN_MM = 4.0

# Ruído utilizado apenas para simular a leitura do sensor.
# O alvo de irrigação NÃO utiliza esse ruído.
SOIL_SENSOR_NOISE_STD = 0.8

# Limite máximo de irrigação por evento.
MAX_IRRIGATION_DEPTH_MM = 8.0


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "weather_santo_amaro_2020_2025_hourly.csv"
OUTPUT_FILE = DATA_DIR / "irrigation_dataset.csv"


# ============================================================
# COEFICIENTE DE CULTIVO DO COENTRO
# Silva, Tavares e Sousa (2013)
# ============================================================

KC_BY_STAGE = {
    "initial": 0.82,
    "development": 1.03,
    "mid": 1.07,
    "late": 0.93,
}


# Fração mínima da capacidade de armazenamento que deve
# permanecer disponível em cada estágio.
STRESS_FRACTION_BY_STAGE = {
    "initial": 0.48,
    "development": 0.45,
    "mid": 0.43,
    "late": 0.46,
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def get_crop_stage(day_of_cycle: int) -> str:
    """
    Retorna o estágio do coentro com base no dia do ciclo.
    """

    if day_of_cycle <= 15:
        return "initial"

    if day_of_cycle <= 30:
        return "development"

    if day_of_cycle <= 42:
        return "mid"

    return "late"


def calculate_crop_cycle(timestamp: pd.Timestamp) -> tuple[int, int]:
    """
    Retorna:
        - número do ciclo
        - dia dentro do ciclo
    """

    day_index = (timestamp.normalize() - df_start_date).days

    cycle_number = day_index // CROP_CYCLE_DAYS + 1
    day_of_cycle = day_index % CROP_CYCLE_DAYS + 1

    return cycle_number, day_of_cycle


def calculate_et0(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estima ET0 diária por uma heurística simplificada.

    IMPORTANTE:
    Este cálculo NÃO é a implementação completa do método
    FAO Penman-Monteith.

    As variáveis utilizadas são:
        - temperatura média
        - umidade relativa média
        - vento médio
        - radiação solar diária
    """

    temperature_factor = np.clip(
        (daily_df["temperature_c"] - 15.0) / 15.0,
        0.0,
        1.0,
    )

    humidity_factor = np.clip(
        (100.0 - daily_df["air_humidity_percent"]) / 100.0,
        0.0,
        1.0,
    )

    wind_factor = np.clip(
        daily_df["wind_speed_m_s"] / 5.0,
        0.0,
        1.0,
    )

    radiation_factor = np.clip(
        daily_df["solar_radiation_kwh_m2"] / 25.0,
        0.0,
        1.0,
    )

    et0 = (
        1.2
        + 2.2 * temperature_factor
        + 1.2 * humidity_factor
        + 0.7 * wind_factor
        + 0.9 * radiation_factor
    )

    daily_df["et0"] = np.clip(et0, 1.0, 7.0)

    return daily_df


# ============================================================
# LEITURA DOS DADOS
# ============================================================

print("Carregando dados meteorológicos...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "timestamp",
    "temperature_c",
    "air_humidity_percent",
    "wind_speed_m_s",
    "solar_radiation_kwh_m2",
    "precipitation_mm",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Colunas ausentes no dataset meteorológico: {missing_columns}"
    )


# ============================================================
# TRATAMENTO TEMPORAL
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
)

if df["timestamp"].isna().any():
    raise ValueError("Existem timestamps inválidos.")

df = df.sort_values("timestamp").reset_index(drop=True)

df_start_date = df["timestamp"].min().normalize()

df["date"] = df["timestamp"].dt.normalize()


# ============================================================
# ET0 DIÁRIA
# ============================================================

print("Calculando ET0 diária...")

daily_weather = (
    df.groupby("date")
    .agg(
        temperature_c=("temperature_c", "mean"),
        air_humidity_percent=("air_humidity_percent", "mean"),
        wind_speed_m_s=("wind_speed_m_s", "mean"),
        solar_radiation_kwh_m2=("solar_radiation_kwh_m2", "sum"),
    )
    .reset_index()
)

daily_weather = calculate_et0(daily_weather)

df = df.merge(
    daily_weather[["date", "et0"]],
    on="date",
    how="left",
)


# ============================================================
# ESTÁGIO DO COENTRO
# ============================================================

print("Calculando estágio da cultura...")

day_index = (
    df["date"] - df_start_date
).dt.days

df["cycle_number"] = (
    day_index // CROP_CYCLE_DAYS + 1
)

df["crop_cycle_day"] = (
    day_index % CROP_CYCLE_DAYS + 1
)

df["crop_stage"] = df["crop_cycle_day"].apply(
    get_crop_stage
)

df["kc"] = df["crop_stage"].map(
    KC_BY_STAGE
)


# ============================================================
# DISTRIBUIÇÃO HORÁRIA DA ETc
# ============================================================

print("Distribuindo ETc ao longo do dia...")

hour = df["timestamp"].dt.hour

solar_profile = np.maximum(
    np.sin(
        np.pi * (hour - 6) / 12
    ),
    0,
)

profile_series = pd.Series(
    solar_profile,
    index=df.index,
)

profile_sum = (
    profile_series
    .groupby(df["date"])
    .transform("sum")
)

hourly_fraction = np.where(
    profile_sum > 0,
    solar_profile / profile_sum,
    0.0,
)

df["etc"] = (
    df["et0"]
    * df["kc"]
    * hourly_fraction
)


# ============================================================
# SIMULAÇÃO DO BALANÇO HÍDRICO
# ============================================================

print("Simulando balanço hídrico e decisões de irrigação...")

rng = np.random.default_rng(RANDOM_SEED)

storage = (
    STORAGE_CAPACITY_MM
    * INITIAL_STORAGE_FRACTION
)

soil_moisture_values = []
dynamic_lower_bound_values = []
irrigation_depth_values = []

storage_before_values = []
storage_after_values = []

for row in df.itertuples():

    # --------------------------------------------------------
    # 1. BALANÇO NATURAL ANTES DA DECISÃO
    # --------------------------------------------------------
    #
    # Primeiro o solo perde água pela ETc e recebe água
    # proveniente da precipitação.
    #
    # Neste momento ainda NÃO houve irrigação.
    # --------------------------------------------------------

    storage -= row.etc

    effective_rain = (
        row.precipitation_mm
        * RAIN_EFFECTIVE_FRACTION
    )

    storage += effective_rain

    storage = np.clip(
        storage,
        0.0,
        STORAGE_CAPACITY_MM,
    )

    # Guardamos o estado real do solo antes da irrigação.
    storage_before_irrigation = storage

    storage_before_values.append(
        storage_before_irrigation
    )

    # --------------------------------------------------------
    # 2. UMIDADE OBSERVADA PELO SENSOR
    # --------------------------------------------------------
    #
    # O sensor observa o estado ANTES da decisão.
    #
    # O ruído representa imperfeição de medição e não altera
    # o estado interno utilizado para calcular o alvo.
    # --------------------------------------------------------

    soil_moisture = (
        storage_before_irrigation
        / STORAGE_CAPACITY_MM
        * 100.0
    )

    sensor_noise = rng.normal(
        0.0,
        SOIL_SENSOR_NOISE_STD,
    )

    soil_moisture_observed = np.clip(
        soil_moisture + sensor_noise,
        0.0,
        100.0,
    )

    soil_moisture_values.append(
        soil_moisture_observed
    )

    # --------------------------------------------------------
    # 3. LIMITE INFERIOR DINÂMICO
    # --------------------------------------------------------

    stress_fraction = (
        STRESS_FRACTION_BY_STAGE[
            row.crop_stage
        ]
    )

    demand_factor = np.clip(
        row.etc / 0.8,
        0.0,
        1.0,
    )

    dynamic_fraction = (
        stress_fraction
        + 0.08 * demand_factor
    )

    dynamic_lower_bound = (
        dynamic_fraction
        * STORAGE_CAPACITY_MM
    )

    dynamic_lower_bound_values.append(
        dynamic_lower_bound
        / STORAGE_CAPACITY_MM
        * 100.0
    )

    # --------------------------------------------------------
    # 4. DECISÃO DE IRRIGAÇÃO
    # --------------------------------------------------------

    irrigation_depth = 0.0

    if storage_before_irrigation < dynamic_lower_bound:

        deficit = (
            dynamic_lower_bound
            - storage_before_irrigation
        )

        # Quantidade líquida necessária para levar o solo
        # acima do limite de segurança.
        required_water = (
            deficit
            + RECOVERY_MARGIN_MM
        )

        irrigation_depth = np.clip(
            required_water,
            0.0,
            MAX_IRRIGATION_DEPTH_MM,
        )

        # ----------------------------------------------------
        # 5. ATUALIZAÇÃO DO ESTADO APÓS A IRRIGAÇÃO
        # ----------------------------------------------------
        #
        # A irrigação armazenada é adicionada ao reservatório
        # somente DEPOIS de termos registrado a entrada
        # soil_moisture_percent.
        # ----------------------------------------------------

        storage += irrigation_depth

        storage = np.clip(
            storage,
            0.0,
            STORAGE_CAPACITY_MM,
        )

    irrigation_depth_values.append(
        irrigation_depth
    )

    storage_after_values.append(
        storage
    )


# ============================================================
# INSERÇÃO DAS VARIÁVEIS GERADAS
# ============================================================

df["soil_moisture_percent"] = (
    soil_moisture_values
)

df["dynamic_lower_bound_percent"] = (
    dynamic_lower_bound_values
)

df["irrigation_depth_mm"] = (
    irrigation_depth_values
)

df["storage_before_irrigation_mm"] = (
    storage_before_values
)

df["storage_after_irrigation_mm"] = (
    storage_after_values
)


# ============================================================
# VARIÁVEIS TEMPORAIS
# ============================================================

print("Calculando variáveis temporais...")

hour_decimal = (
    df["timestamp"].dt.hour
    + df["timestamp"].dt.minute / 60.0
)

df["hour_sin"] = np.sin(
    2 * np.pi * hour_decimal / 24.0
)

df["hour_cos"] = np.cos(
    2 * np.pi * hour_decimal / 24.0
)


# ============================================================
# ORGANIZAÇÃO FINAL
# ============================================================

df.insert(
    0,
    "id",
    np.arange(1, len(df) + 1),
)


final_columns = [
    "id",
    "timestamp",

    "temperature_c",
    "air_humidity_percent",
    "wind_speed_m_s",
    "solar_radiation_kwh_m2",
    "precipitation_mm",

    "soil_moisture_percent",

    "crop_stage",
    "kc",

    "et0",
    "etc",

    "dynamic_lower_bound_percent",

    "irrigation_depth_mm",

    "hour_sin",
    "hour_cos",
]


df = df[final_columns]


# ============================================================
# VALIDAÇÕES
# ============================================================

print("Validando dataset...")

if df.isna().any().any():
    missing = df.isna().sum()
    missing = missing[missing > 0]

    raise ValueError(
        f"Existem valores ausentes:\n{missing}"
    )


if df["timestamp"].duplicated().any():
    raise ValueError(
        "Existem timestamps duplicados."
    )


if (df["etc"] < 0).any():
    raise ValueError(
        "ETc contém valores negativos."
    )


if (df["irrigation_depth_mm"] < 0).any():
    raise ValueError(
        "Irrigação contém valores negativos."
    )


if (
    df["irrigation_depth_mm"]
    > MAX_IRRIGATION_DEPTH_MM
).any():
    raise ValueError(
        "Existem eventos acima do limite máximo."
    )


if df["irrigation_depth_mm"].sum() <= 0:
    raise ValueError(
        "Nenhum evento de irrigação foi gerado."
    )


# ============================================================
# SALVAMENTO
# ============================================================

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# RESUMO
# ============================================================

irrigation_events = (
    df["irrigation_depth_mm"] > 0
)

number_of_events = (
    irrigation_events.sum()
)

event_rate = (
    number_of_events
    / len(df)
    * 100
)

event_depths = (
    df.loc[
        irrigation_events,
        "irrigation_depth_mm"
    ]
)

print()
print("=" * 60)
print("DATASET V4 GERADO COM SUCESSO")
print("=" * 60)

print(f"Arquivo: {OUTPUT_FILE}")
print(f"Registros: {len(df):,}")
print(
    f"Período: "
    f"{df['timestamp'].min()} → "
    f"{df['timestamp'].max()}"
)

print()
print("Eventos de irrigação:")
print(f"  Quantidade: {number_of_events:,}")
print(f"  Taxa:       {event_rate:.2f}%")

print()
print("Irrigação por evento:")
print(f"  Média:   {event_depths.mean():.3f} mm")
print(f"  Mediana: {event_depths.median():.3f} mm")
print(f"  Mínima:  {event_depths.min():.3f} mm")
print(f"  Máxima:  {event_depths.max():.3f} mm")

print()
print("Umidade do solo:")
print(
    f"  Média:   "
    f"{df['soil_moisture_percent'].mean():.3f}%"
)
print(
    f"  Mínima:  "
    f"{df['soil_moisture_percent'].min():.3f}%"
)
print(
    f"  Máxima:  "
    f"{df['soil_moisture_percent'].max():.3f}%"
)

print()
print("ET0:")
print(
    f"  Média:   "
    f"{df['et0'].mean():.3f} mm/dia"
)

print()
print("ETc:")
print(
    f"  Média:   "
    f"{df['etc'].mean():.3f} mm/h"
)

print()
print("Estágios:")
print(
    df["crop_stage"]
    .value_counts()
    .sort_index()
)

print()
print("Dataset salvo em:")
print(OUTPUT_FILE)

print("=" * 60)