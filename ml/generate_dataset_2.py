from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# IRRIGATION DATASET 2
# ============================================================
# Gera um dataset ampliado a partir do clima REAL da NASA POWER.
#
# Ideia:
# - não duplica simplesmente as mesmas linhas;
# - simula vários ciclos de coentro sobre diferentes janelas
#   meteorológicas e diferentes condições iniciais de solo;
# - mantém a decisão de irrigação horária;
# - mantém o estágio da planta e Kc;
# - corrige a umidade para representar o estado ANTES da
#   decisão de irrigação;
# - gera features temporais/lag para dar contexto ao ML.
#
# Saída:
#   data/irrigation_dataset_2.csv
#
# IMPORTANTE:
# O arquivo de clima precisa estar em:
#   data/weather_santo_amaro_2020_2025_hourly.csv
#
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "weather_santo_amaro_2020_2025_hourly.csv"
OUTPUT_FILE = DATA_DIR / "irrigation_dataset_2.csv"

RANDOM_SEED = 42

# ============================================================
# CULTURA
# ============================================================

CROP_CYCLE_DAYS = 50

KC_BY_STAGE = {
    "initial": 0.82,
    "development": 1.03,
    "mid": 1.07,
    "late": 0.93,
}

STRESS_FRACTION = {
    "initial": 0.48,
    "development": 0.45,
    "mid": 0.43,
    "late": 0.46,
}

# ============================================================
# CENÁRIOS
# ============================================================
# Cada cenário representa uma condição diferente do reservatório
# hídrico simulado.
#
# A capacidade é uma escala de água disponível no sistema, NÃO
# uma medição direta de um solo específico.
#
# A variação é propositalmente controlada, não aleatória.
# ============================================================

SCENARIOS = [
    {"scenario_id": 1, "storage_capacity_mm": 40.0, "initial_fraction": 0.55},
    {"scenario_id": 2, "storage_capacity_mm": 40.0, "initial_fraction": 0.70},
    {"scenario_id": 3, "storage_capacity_mm": 45.0, "initial_fraction": 0.60},
    {"scenario_id": 4, "storage_capacity_mm": 45.0, "initial_fraction": 0.80},
    {"scenario_id": 5, "storage_capacity_mm": 50.0, "initial_fraction": 0.65},
    {"scenario_id": 6, "storage_capacity_mm": 50.0, "initial_fraction": 0.80},
    {"scenario_id": 7, "storage_capacity_mm": 55.0, "initial_fraction": 0.70},
    {"scenario_id": 8, "storage_capacity_mm": 60.0, "initial_fraction": 0.70},
    {"scenario_id": 9, "storage_capacity_mm": 65.0, "initial_fraction": 0.75},
    {"scenario_id": 10, "storage_capacity_mm": 70.0, "initial_fraction": 0.75},
    {"scenario_id": 11, "storage_capacity_mm": 75.0, "initial_fraction": 0.80},
    {"scenario_id": 12, "storage_capacity_mm": 80.0, "initial_fraction": 0.80},
]

RAIN_EFFECTIVE_FRACTION = 0.75
RECOVERY_MARGIN_MM = 4.0
SOIL_SENSOR_NOISE_STD = 0.8

# Margem fixa: evita colocar ruído aleatório diretamente no alvo.
# O alvo continua representando uma política determinística.
IRRIGATION_RECOVERY_MM = 4.0

MAX_IRRIGATION_DEPTH_MM = 8.0

# ============================================================
# ESTÁGIO
# ============================================================

def calculate_crop_stage(day_of_cycle):
    if day_of_cycle <= 15:
        return "initial"
    if day_of_cycle <= 30:
        return "development"
    if day_of_cycle <= 42:
        return "mid"
    return "late"


def add_crop_information(df):
    first_date = df["timestamp"].min().normalize()

    df["days_since_start"] = (
        df["timestamp"].dt.normalize() - first_date
    ).dt.days

    df["day_of_cycle"] = (
        df["days_since_start"] % CROP_CYCLE_DAYS
    ) + 1

    df["crop_stage"] = df["day_of_cycle"].apply(
        calculate_crop_stage
    )

    df["kc"] = df["crop_stage"].map(KC_BY_STAGE)

    return df


# ============================================================
# ET0
# ============================================================

def calculate_daily_et0(df):
    """
    Estimativa simplificada de ET0.

    ATENÇÃO:
    Não é FAO Penman-Monteith.

    A estimativa é usada como variável derivada para a simulação.
    """

    df["date"] = df["timestamp"].dt.date

    daily = (
        df.groupby("date")
        .agg(
            temperature_mean=("temperature_c", "mean"),
            temperature_max=("temperature_c", "max"),
            temperature_min=("temperature_c", "min"),
            humidity_mean=("air_humidity_percent", "mean"),
            wind_mean=("wind_speed_m_s", "mean"),
            radiation_sum=("solar_radiation_kwh_m2", "sum"),
        )
        .reset_index()
    )

    temperature_factor = np.clip(
        (daily["temperature_mean"] - 15) / 15,
        0,
        1,
    )

    humidity_factor = np.clip(
        (100 - daily["humidity_mean"]) / 100,
        0,
        1,
    )

    wind_factor = np.clip(
        daily["wind_mean"] / 5,
        0,
        1,
    )

    radiation_factor = np.clip(
        daily["radiation_sum"] / 25,
        0,
        1,
    )

    et0 = (
        1.2
        + 2.2 * temperature_factor
        + 1.2 * humidity_factor
        + 0.7 * wind_factor
        + 0.9 * radiation_factor
    )

    daily["et0"] = np.clip(
        et0,
        1.0,
        7.0,
    )

    return df.merge(
        daily[["date", "et0"]],
        on="date",
        how="left",
    )


# ============================================================
# ETc HORÁRIA
# ============================================================

def calculate_hourly_etc(df):
    hour = df["timestamp"].dt.hour

    solar_profile = np.maximum(
        np.sin(np.pi * (hour - 6) / 12),
        0,
    )

    profile_sum = (
        pd.Series(
            solar_profile,
            index=df.index,
        )
        .groupby(df["date"])
        .transform("sum")
    )

    hourly_fraction = np.where(
        profile_sum > 0,
        solar_profile / profile_sum,
        0,
    )

    df["etc"] = (
        df["et0"]
        * df["kc"]
        * hourly_fraction
    )

    return df


# ============================================================
# SIMULAÇÃO DE UM CENÁRIO
# ============================================================

def simulate_scenario(base_df, scenario):
    """
    Simula uma trajetória hídrica independente.

    A sequência temporal é:

        estado anterior
            ↓
        chuva efetiva
            ↓
        ETc
            ↓
        estado pré-irrigação
            ↓
        sensor de umidade
            ↓
        decisão de irrigação
            ↓
        aplicação
            ↓
        próximo estado
    """

    df = base_df.copy()

    capacity = scenario["storage_capacity_mm"]
    water_storage = (
        capacity * scenario["initial_fraction"]
    )

    recently_irrigated = False

    soil_moisture = []
    lower_bound = []
    irrigation = []
    storage_before_irrigation = []
    storage_after_irrigation = []

    rng = np.random.default_rng(
        RANDOM_SEED + scenario["scenario_id"] * 1000
    )

    for _, row in df.iterrows():

        stage = row["crop_stage"]

        # ----------------------------------------------------
        # 1. CHUVA
        # ----------------------------------------------------

        effective_rain = (
            row["precipitation_mm"]
            * RAIN_EFFECTIVE_FRACTION
        )

        water_storage += effective_rain

        # ----------------------------------------------------
        # 2. ETc
        # ----------------------------------------------------

        water_storage -= row["etc"]

        water_storage = max(
            water_storage,
            0.0,
        )

        water_storage = min(
            water_storage,
            capacity,
        )

        # ----------------------------------------------------
        # 3. LIMITE DINÂMICO
        # ----------------------------------------------------

        base_fraction = STRESS_FRACTION[stage]

        demand_factor = np.clip(
            row["etc"] / 0.8,
            0,
            1,
        )

        dynamic_fraction = (
            base_fraction
            + 0.08 * demand_factor
        )

        minimum_storage = (
            capacity
            * dynamic_fraction
        )

        # ----------------------------------------------------
        # 4. ESTADO PRÉ-IRRIGAÇÃO
        # ----------------------------------------------------

        storage_before = water_storage

        moisture_percent = (
            storage_before
            / capacity
            * 100
        )

        moisture_percent += rng.normal(
            0,
            SOIL_SENSOR_NOISE_STD,
        )

        moisture_percent = np.clip(
            moisture_percent,
            0,
            100,
        )

        # ----------------------------------------------------
        # 5. DECISÃO
        # ----------------------------------------------------

        trigger_level = minimum_storage

        if recently_irrigated:
            trigger_level = (
                minimum_storage
                - RECOVERY_MARGIN_MM
            )

        irrigation_depth = 0.0

        if storage_before < trigger_level:

            deficit = (
                minimum_storage
                - storage_before
            )

            # Reposição fixa, sem ruído no alvo.
            irrigation_depth = (
                deficit
                + IRRIGATION_RECOVERY_MM
            )

            irrigation_depth = np.clip(
                irrigation_depth,
                0,
                MAX_IRRIGATION_DEPTH_MM,
            )

            water_storage += irrigation_depth

            water_storage = min(
                water_storage,
                capacity,
            )

            recently_irrigated = True

        else:

            if water_storage > (
                minimum_storage
                + RECOVERY_MARGIN_MM
            ):
                recently_irrigated = False

        # ----------------------------------------------------
        # 6. SALVAR
        # ----------------------------------------------------

        soil_moisture.append(
            moisture_percent
        )

        lower_bound.append(
            dynamic_fraction * 100
        )

        irrigation.append(
            irrigation_depth
        )

        storage_before_irrigation.append(
            storage_before
        )

        storage_after_irrigation.append(
            water_storage
        )

    df["scenario_id"] = scenario["scenario_id"]

    df["storage_capacity_mm"] = capacity

    df["soil_moisture_percent"] = soil_moisture

    df["dynamic_lower_bound_percent"] = lower_bound

    df["irrigation_depth_mm"] = irrigation

    df["storage_before_irrigation_mm"] = (
        storage_before_irrigation
    )

    df["storage_after_irrigation_mm"] = (
        storage_after_irrigation
    )

    return df


# ============================================================
# FEATURES TEMPORAIS
# ============================================================

def add_time_features(df):

    hour = df["timestamp"].dt.hour

    df["hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    # Dia do ano
    day_of_year = (
        df["timestamp"].dt.dayofyear
    )

    df["day_of_year_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    df["day_of_year_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    return df


# ============================================================
# FEATURES DE MEMÓRIA
# ============================================================

def add_lag_features(df):
    """
    Features disponíveis no instante t.

    Todas são calculadas a partir do histórico anterior.
    """

    grouped = df.groupby("scenario_id", sort=False)

    # Umidade observada anteriormente
    df["soil_moisture_lag_1h"] = (
        grouped["soil_moisture_percent"]
        .shift(1)
    )

    df["soil_moisture_lag_3h"] = (
        grouped["soil_moisture_percent"]
        .shift(3)
    )

    df["soil_moisture_lag_6h"] = (
        grouped["soil_moisture_percent"]
        .shift(6)
    )

    # ETc acumulada
    df["etc_6h"] = (
        grouped["etc"]
        .transform(
            lambda s: s.rolling(6, min_periods=1).sum()
        )
    )

    df["etc_24h"] = (
        grouped["etc"]
        .transform(
            lambda s: s.rolling(24, min_periods=1).sum()
        )
    )

    # Chuva acumulada.
    # Shift(1) evita usar a chuva do próprio instante
    # como parte do histórico.
    rain_previous = grouped["precipitation_mm"].shift(1)

    df["rain_6h"] = (
        rain_previous
        .groupby(df["scenario_id"])
        .transform(
            lambda s: s.rolling(6, min_periods=1).sum()
        )
        .fillna(0)
    )

    df["rain_24h"] = (
        rain_previous
        .groupby(df["scenario_id"])
        .transform(
            lambda s: s.rolling(24, min_periods=1).sum()
        )
        .fillna(0)
    )

    # Temperatura média recente
    df["temperature_6h_mean"] = (
        grouped["temperature_c"]
        .transform(
            lambda s: s.rolling(6, min_periods=1).mean()
        )
    )

    # Radiação acumulada
    df["radiation_6h_sum"] = (
        grouped["solar_radiation_kwh_m2"]
        .transform(
            lambda s: s.rolling(6, min_periods=1).sum()
        )
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("IRRIGATION DATASET 2")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nArquivo não encontrado:\n{INPUT_FILE}\n\n"
            "Coloque weather_santo_amaro_2020_2025_hourly.csv "
            "na pasta data/."
        )

    print("\nCarregando NASA POWER...")

    weather = pd.read_csv(INPUT_FILE)

    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"]
    )

    weather = (
        weather
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        f"Registros meteorológicos: "
        f"{len(weather):,}"
    )

    # --------------------------------------------------------
    # Cultura
    # --------------------------------------------------------

    print("Calculando estágios do coentro...")

    weather = add_crop_information(weather)

    # --------------------------------------------------------
    # ET0
    # --------------------------------------------------------

    print("Calculando ET0...")

    weather = calculate_daily_et0(weather)

    # --------------------------------------------------------
    # ETc
    # --------------------------------------------------------

    print("Calculando ETc...")

    weather = calculate_hourly_etc(weather)

    # --------------------------------------------------------
    # Cenários
    # --------------------------------------------------------

    print(
        f"\nGerando {len(SCENARIOS)} "
        f"cenários hídricos..."
    )

    scenario_frames = []

    for scenario in SCENARIOS:

        print(
            f"  Cenário {scenario['scenario_id']:02d} "
            f"| capacidade="
            f"{scenario['storage_capacity_mm']:.0f} mm "
            f"| inicial="
            f"{scenario['initial_fraction'] * 100:.0f}%"
        )

        simulated = simulate_scenario(
            weather,
            scenario,
        )

        scenario_frames.append(
            simulated
        )

    df = pd.concat(
        scenario_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    print("\nCriando features temporais...")

    df = add_time_features(df)

    print("Criando features de memória...")

    df = add_lag_features(df)

    # --------------------------------------------------------
    # Remover primeiras linhas de cada cenário
    # --------------------------------------------------------
    # As primeiras horas não possuem todo o histórico necessário
    # para os lags. Removê-las evita preencher artificialmente.
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "soil_moisture_lag_6h",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    df.insert(
        0,
        "id",
        range(
            1,
            len(df) + 1,
        ),
    )

    # --------------------------------------------------------
    # Ordenação
    # --------------------------------------------------------

    df = (
        df
        .sort_values(
            ["timestamp", "scenario_id"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Colunas finais
    # --------------------------------------------------------

    columns = [
        "id",
        "timestamp",
        "scenario_id",
        "storage_capacity_mm",

        "temperature_c",
        "air_humidity_percent",
        "wind_speed_m_s",
        "solar_radiation_kwh_m2",
        "precipitation_mm",

        "soil_moisture_percent",
        "soil_moisture_lag_1h",
        "soil_moisture_lag_3h",
        "soil_moisture_lag_6h",

        "crop_stage",
        "kc",

        "et0",
        "etc",
        "etc_6h",
        "etc_24h",

        "rain_6h",
        "rain_24h",

        "temperature_6h_mean",
        "radiation_6h_sum",

        "dynamic_lower_bound_percent",

        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",

        "irrigation_depth_mm",

        "storage_before_irrigation_mm",
        "storage_after_irrigation_mm",
    ]

    df = df[columns]

    # --------------------------------------------------------
    # Salvar
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Estatísticas
    # --------------------------------------------------------

    events = df[
        df["irrigation_depth_mm"] > 0
    ]

    print()
    print("=" * 70)
    print("DATASET 2 GERADO")
    print("=" * 70)

    print(
        f"Registros: {len(df):,}"
    )

    print(
        f"Cenários: {df['scenario_id'].nunique()}"
    )

    print(
        f"Período: "
        f"{df['timestamp'].min()} "
        f"→ "
        f"{df['timestamp'].max()}"
    )

    print(
        f"\nEventos de irrigação: "
        f"{len(events):,}"
    )

    print(
        f"Percentual de eventos: "
        f"{len(events) / len(df) * 100:.2f}%"
    )

    if len(events):

        print(
            f"Média por evento: "
            f"{events['irrigation_depth_mm'].mean():.3f} mm"
        )

        print(
            f"Mediana por evento: "
            f"{events['irrigation_depth_mm'].median():.3f} mm"
        )

        print(
            f"Máximo por evento: "
            f"{events['irrigation_depth_mm'].max():.3f} mm"
        )

        print(
            f"Água total: "
            f"{df['irrigation_depth_mm'].sum():.2f} mm"
        )

    print(
        f"\nUmidade mínima: "
        f"{df['soil_moisture_percent'].min():.2f}%"
    )

    print(
        f"Umidade média: "
        f"{df['soil_moisture_percent'].mean():.2f}%"
    )

    print(
        f"Umidade máxima: "
        f"{df['soil_moisture_percent'].max():.2f}%"
    )

    # Eventos por cenário
    print("\nEVENTOS POR CENÁRIO")

    events_by_scenario = (
        df.assign(
            event=df["irrigation_depth_mm"] > 0
        )
        .groupby("scenario_id")["event"]
        .sum()
    )

    print(events_by_scenario.to_string())

    print(
        f"\nArquivo salvo em:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
