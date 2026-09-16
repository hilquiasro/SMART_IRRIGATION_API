"""
generate_dataset_3.py

Dataset 3 — versão consolidada para o projeto de irrigação inteligente.

Entrada:
    ml/data/weather_santo_amaro_2020_2025_hourly.csv

Saída:
    ml/data/irrigation_dataset_3.csv

Metodologia:
- Meteorologia real do NASA POWER.
- ET0 estimada por uma heurística simplificada (não é FAO Penman-Monteith).
- ETc = ET0 * Kc do coentro.
- Balanço hídrico horário para simular a umidade antes da decisão.
- Múltiplos cenários hídricos para aumentar diversidade.
- Target = lâmina de irrigação (mm).
- Sem uso de aleatoriedade para a chuva efetiva ou para o target.
- Ruído apenas na leitura simulada do sensor de umidade.
- O target não é limitado artificialmente a 8 mm; há um limite operacional
  mais alto apenas para impedir valores absurdos.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURAÇÃO
# ============================================================

RANDOM_SEED = 42
CROP_CYCLE_DAYS = 50

# Limite físico-operacional da lâmina por evento.
# O Dataset 2 mostrou que 8 mm estava truncando muitos eventos.
MAX_IRRIGATION_DEPTH_MM = 15.0

# Entrada de chuva efetiva no balanço.
RAIN_EFFECTIVE_FRACTION = 0.75

# Ruído do sensor simulado.
SOIL_SENSOR_NOISE_STD = 0.8

# Margem adicionada após repor o déficit.
RECOVERY_MARGIN_MM = 4.0

# Cenários: capacidade de armazenamento, armazenamento inicial,
# fração-base de acionamento, margem extra por demanda e fração
# efetiva da chuva.
#
# Os cenários representam diferentes condições de retenção/solo e
# estratégias hídricas plausíveis, não novas medições meteorológicas.
SCENARIOS = [
    # id, capacidade_mm, inicial_%, stress_fraction, demand_gain, rain_eff
    (1,  35.0, 55.0, 0.44, 0.06, 0.70),
    (2,  35.0, 70.0, 0.47, 0.06, 0.80),
    (3,  40.0, 60.0, 0.43, 0.07, 0.70),
    (4,  40.0, 80.0, 0.46, 0.07, 0.80),
    (5,  45.0, 60.0, 0.42, 0.08, 0.75),
    (6,  45.0, 75.0, 0.46, 0.08, 0.85),
    (7,  50.0, 65.0, 0.42, 0.08, 0.70),
    (8,  50.0, 80.0, 0.46, 0.08, 0.85),
    (9,  55.0, 70.0, 0.41, 0.09, 0.75),
    (10, 60.0, 70.0, 0.43, 0.09, 0.80),
    (11, 65.0, 75.0, 0.44, 0.09, 0.85),
    (12, 70.0, 80.0, 0.45, 0.10, 0.80),
    (13, 75.0, 80.0, 0.45, 0.10, 0.85),
    (14, 85.0, 80.0, 0.46, 0.10, 0.85),
]

KC_BY_STAGE = {
    "initial": 0.82,
    "development": 1.03,
    "mid": 1.07,
    "late": 0.93,
}

BASE_STRESS_FRACTION = {
    "initial": 0.48,
    "development": 0.45,
    "mid": 0.43,
    "late": 0.46,
}


# ============================================================
# FUNÇÕES
# ============================================================

def crop_stage(day_of_cycle: int) -> str:
    if day_of_cycle <= 15:
        return "initial"
    if day_of_cycle <= 30:
        return "development"
    if day_of_cycle <= 42:
        return "mid"
    return "late"


def calculate_et0(df: pd.DataFrame) -> pd.Series:
    """
    Estimativa simplificada de ET0 diária.
    """
    # Criamos uma coluna de data explicitamente para o agrupamento
    temp_df = df.copy()
    temp_df["date"] = temp_df["timestamp"].dt.floor("D")

    daily = (
        temp_df.groupby("date")
        .agg(
            mean_temp=("temperature_c", "mean"),
            mean_rh=("air_humidity_percent", "mean"),
            mean_wind=("wind_speed_m_s", "mean"),
            radiation_sum=("solar_radiation_kwh_m2", "sum"),
        )
    )

    temperature_factor = np.clip((daily["mean_temp"] - 15.0) / 15.0, 0.0, 1.0)
    humidity_factor = np.clip((100.0 - daily["mean_rh"]) / 100.0, 0.0, 1.0)
    wind_factor = np.clip(daily["mean_wind"] / 5.0, 0.0, 1.0)
    radiation_factor = np.clip(daily["radiation_sum"] / 25.0, 0.0, 1.0)

    et0 = (
        1.2
        + 2.2 * temperature_factor
        + 1.2 * humidity_factor
        + 0.7 * wind_factor
        + 0.9 * radiation_factor
    )

    return pd.Series(
        np.clip(et0, 1.0, 7.0),
        index=daily.index,
        name="et0",
    )


def build_base_weather(weather: pd.DataFrame) -> pd.DataFrame:
    weather = weather.copy()

    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    weather = weather.sort_values("timestamp").drop_duplicates("timestamp")

    weather["day_of_year"] = weather["timestamp"].dt.dayofyear
    weather["hour"] = weather["timestamp"].dt.hour
    weather["date"] = weather["timestamp"].dt.floor("D")

    # Estágio do coentro em ciclos de 50 dias
    day_index = (
        weather["date"] - weather["date"].min()
    ).dt.days

    cycle_day = (day_index % CROP_CYCLE_DAYS) + 1
    weather["crop_stage"] = cycle_day.map(crop_stage)
    weather["kc"] = weather["crop_stage"].map(KC_BY_STAGE)

    # Cálculo da ET0 diária e mapeamento seguro via merge
    daily_et0 = calculate_et0(weather).reset_index()
    
    # Merge direto evita perda de dados por desencontro de índices
    weather = weather.merge(daily_et0, on="date", how="left")

    # ETc diária e distribuição horária
    weather["etc_daily"] = weather["et0"] * weather["kc"]

    # distribute_etc_hourly precisa de uma série indexada por DATA
    # (uma linha por dia), não pelo índice padrão de linhas de 'weather'.
    daily_etc_by_date = (
        weather.drop_duplicates("date").set_index("date")["etc_daily"]
    )

    weather["etc"] = distribute_etc_hourly(
        daily_etc_by_date,
        weather["timestamp"],
    )
    # Limpeza da coluna auxiliar 'date'
    weather = weather.drop(columns=["date"])

    # Codificação cíclica
    weather["hour_sin"] = np.sin(2 * np.pi * weather["hour"] / 24.0)
    weather["hour_cos"] = np.cos(2 * np.pi * weather["hour"] / 24.0)

    weather["day_of_year_sin"] = np.sin(
        2 * np.pi * weather["day_of_year"] / 365.25
    )
    weather["day_of_year_cos"] = np.cos(
        2 * np.pi * weather["day_of_year"] / 365.25
    )

    # Verificação preventiva: garante que a base weather não tem NaN em etc
    if weather["etc"].isna().any():
        raise ValueError(
            f"Falha na construção do clima base: {weather['etc'].isna().sum()} linhas com 'etc' nulo."
        )

    return weather

def distribute_etc_hourly(daily_etc: pd.Series, timestamps: pd.Series) -> np.ndarray:
    """
    Distribui a ETc diária entre as horas usando um perfil de luz simples.
    O perfil é normalizado para que a soma diária seja exatamente ETc.
    """
    hours = timestamps.dt.hour.to_numpy()

    # Perfil de demanda entre aproximadamente 06h e 18h.
    daylight = np.sin(np.pi * (hours - 6) / 12)
    daylight = np.clip(daylight, 0.0, None)

    # Para evitar dia sem massa numérica por arredondamentos.
    profile = daylight + 1e-9

    dates = timestamps.dt.floor("D")
    weights = pd.Series(profile, index=timestamps.index)

    sums = weights.groupby(dates).transform("sum").to_numpy()

    etc_hourly = weights.to_numpy() / sums

    daily_values = dates.map(daily_etc).to_numpy()

    return etc_hourly * daily_values


def simulate_scenario(base: pd.DataFrame, scenario) -> pd.DataFrame:
    (
        scenario_id,
        storage_capacity,
        initial_percent,
        stress_fraction,
        demand_gain,
        rain_efficiency,
    ) = scenario

    df = base.copy().reset_index(drop=True)

    storage = storage_capacity * initial_percent / 100.0

    soil_values = []
    irrigation_values = []
    storage_before_values = []
    storage_after_values = []
    lower_bound_values = []

    rng = np.random.default_rng(RANDOM_SEED + scenario_id)

    for i in range(len(df)):
        etc = float(df.at[i, "etc"])
        rain = float(df.at[i, "precipitation_mm"])

        # 1. Perda por evapotranspiração.
        storage -= etc

        # 2. Entrada de chuva efetiva.
        effective_rain = rain * rain_efficiency * RAIN_EFFECTIVE_FRACTION
        storage += effective_rain

        storage = float(np.clip(storage, 0.0, storage_capacity))

        # Estado ANTES da irrigação.
        storage_before = storage

        # 3. Sensor mede o estado pré-decisão.
        ideal_moisture = 100.0 * storage_before / storage_capacity
        measured_moisture = ideal_moisture + rng.normal(
            0.0,
            SOIL_SENSOR_NOISE_STD,
        )
        measured_moisture = float(np.clip(measured_moisture, 0.0, 100.0))

        # 4. Limite dinâmico de segurança hídrica.
        stage = df.at[i, "crop_stage"]
        base_fraction = BASE_STRESS_FRACTION[stage]

        # Demanda relativa aumenta a margem de segurança em dias de maior ETc.
        demand_factor = float(np.clip(etc / 0.8, 0.0, 1.0))
        dynamic_fraction = base_fraction + demand_gain * demand_factor

        minimum_storage = storage_capacity * dynamic_fraction

        # 5. Decisão.
        deficit = max(0.0, minimum_storage - storage_before)

        if deficit > 0:
            irrigation = deficit + RECOVERY_MARGIN_MM
            irrigation = min(irrigation, MAX_IRRIGATION_DEPTH_MM)
        else:
            irrigation = 0.0

        # 6. Irrigação ocorre DEPOIS da observação e do target.
        storage_after = float(
            np.clip(storage_before + irrigation, 0.0, storage_capacity)
        )

        soil_values.append(measured_moisture)
        irrigation_values.append(float(irrigation))
        storage_before_values.append(storage_before)
        storage_after_values.append(storage_after)
        lower_bound_values.append(
            100.0 * minimum_storage / storage_capacity
        )

        storage = storage_after

    df["scenario_id"] = scenario_id
    df["storage_capacity_mm"] = storage_capacity
    df["soil_moisture_percent"] = soil_values
    df["irrigation_depth_mm"] = irrigation_values
    df["storage_before_irrigation_mm"] = storage_before_values
    df["storage_after_irrigation_mm"] = storage_after_values
    df["dynamic_lower_bound_percent"] = lower_bound_values

    return df


# ============================================================
# FEATURES DE MEMÓRIA (DIAGNÓSTICO E CORREÇÃO)
# ============================================================

def add_memory_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["scenario_id", "timestamp"]).reset_index(drop=True)

    grouped = df.groupby("scenario_id", group_keys=False)

    # 1. Lags simples
    for lag in [1, 3, 6]:
        df[f"soil_moisture_lag_{lag}h"] = grouped["soil_moisture_percent"].shift(lag)

    # 2. Rolling Window (usando transformação direta)
    df["etc_6h"] = grouped["etc"].transform(lambda x: x.rolling(6, min_periods=6).sum())
    df["etc_24h"] = grouped["etc"].transform(lambda x: x.rolling(24, min_periods=24).sum())

    df["rain_6h"] = grouped["precipitation_mm"].transform(lambda x: x.rolling(6, min_periods=6).sum())
    df["rain_24h"] = grouped["precipitation_mm"].transform(lambda x: x.rolling(24, min_periods=24).sum())

    df["temperature_6h_mean"] = grouped["temperature_c"].transform(lambda x: x.rolling(6, min_periods=6).mean())
    df["radiation_6h_sum"] = grouped["solar_radiation_kwh_m2"].transform(lambda x: x.rolling(6, min_periods=6).sum())

    memory_columns = [
        "soil_moisture_lag_1h",
        "soil_moisture_lag_3h",
        "soil_moisture_lag_6h",
        "etc_6h",
        "etc_24h",
        "rain_6h",
        "rain_24h",
        "temperature_6h_mean",
        "radiation_6h_sum",
    ]

    # --- DIAGNÓSTICO DETALHADO ---
    print("\n--- DIAGNÓSTICO DAS FEATURES DE MEMÓRIA ---")
    has_error = False
    for col in memory_columns:
        valid_count = df[col].notna().sum()
        total_count = len(df)
        print(f"Coluna: {col:<25} | Válidos: {valid_count:,} / {total_count:,}")
        if valid_count == 0:
            has_error = True
            print(f"  ❌ ALERTA: A coluna '{col}' está 100% vazia (NaN)!")

    # Verificar dados meteorológicos de entrada
    input_cols = ["etc", "precipitation_mm", "temperature_c", "solar_radiation_kwh_m2", "soil_moisture_percent"]
    for col in input_cols:
        if col in df.columns and df[col].isna().any():
            print(f"  ⚠️ CUIDADO: A coluna de entrada '{col}' possui {df[col].isna().sum()} valores NaN antes do rolling!")

    if has_error:
        raise RuntimeError("Uma ou mais colunas de memória ficaram completamente com NaN. Verifique o log acima.")

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    np.random.seed(RANDOM_SEED)

    root = Path(__file__).resolve().parents[1]
    input_path = root / "ml" / "data" / "weather_santo_amaro_2020_2025_hourly.csv"
    output_path = root / "ml" / "data" / "irrigation_dataset_3.csv"

    print("=" * 70)
    print("IRRIGATION DATASET 3")
    print("=" * 70)
    print()
    print("Carregando NASA POWER...")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{input_path}"
        )

    weather = pd.read_csv(input_path, parse_dates=["timestamp"])

    required = [
        "timestamp",
        "temperature_c",
        "air_humidity_percent",
        "wind_speed_m_s",
        "solar_radiation_kwh_m2",
        "precipitation_mm",
    ]

    missing = [c for c in required if c not in weather.columns]
    if missing:
        raise ValueError(
            f"Colunas ausentes no arquivo meteorológico: {missing}"
        )

    weather = weather[required].copy()

    print(f"Registros meteorológicos: {len(weather):,}")
    print("Calculando estágios do coentro...")
    print("Calculando ET0...")
    print("Calculando ETc...")
    print()
    print(f"Gerando {len(SCENARIOS)} cenários hídricos...")

    base = build_base_weather(weather)

    scenario_frames = []

    for scenario in SCENARIOS:
        (
            scenario_id,
            capacity,
            initial,
            stress,
            demand_gain,
            rain_eff,
        ) = scenario

        print(
            f"  Cenário {scenario_id:02d} | "
            f"capacidade={capacity:g} mm | "
            f"inicial={initial:g}% | "
            f"chuva={rain_eff:.2f}"
        )

        scenario_frames.append(
            simulate_scenario(base, scenario)
        )

    df = pd.concat(
        scenario_frames,
        ignore_index=True,
    )

    print()
    print("Criando features temporais...")
    print("Criando features de memória...")

    df = add_memory_features(df)

    # Remover somente as primeiras linhas necessárias para os lags.
    before_drop = len(df)

    df = df.dropna(
        subset=[
            "soil_moisture_lag_6h",
            "etc_24h",
            "rain_24h",
            "temperature_6h_mean",
            "radiation_6h_sum",
        ]
    ).reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(
            "Todas as linhas foram removidas pelo dropna(). "
            f"Antes: {before_drop:,}; depois: 0. "
            "As features de memória não foram calculadas corretamente."
        )

    # Colunas finais.
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

    # IDs globais únicos.
    df.insert(
        0,
        "_row_id",
        np.arange(1, len(df) + 1),
    )
    df["id"] = df["_row_id"]
    df = df.drop(columns=["_row_id"])

    df = df[columns]

    # Ordenação temporal por cenário.
    df = df.sort_values(
        ["timestamp", "scenario_id"]
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # ========================================================
    # AUDITORIA
    # ========================================================

    events = df[df["irrigation_depth_mm"] > 0]

    print()
    print("=" * 70)
    print("DATASET 3 GERADO")
    print("=" * 70)
    print(f"Registros: {len(df):,}")
    print(f"Cenários: {df['scenario_id'].nunique()}")
    print(
        f"Período: {df['timestamp'].min()} → "
        f"{df['timestamp'].max()}"
    )
    print()
    print(f"Eventos de irrigação: {len(events):,}")
    event_rate = (100 * len(events) / len(df)) if len(df) else 0.0
    print(
        f"Percentual de eventos: "
        f"{event_rate:.3f}%"
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
            f"P90 por evento: "
            f"{events['irrigation_depth_mm'].quantile(.90):.3f} mm"
        )
        print(
            f"P95 por evento: "
            f"{events['irrigation_depth_mm'].quantile(.95):.3f} mm"
        )
        print(
            f"P99 por evento: "
            f"{events['irrigation_depth_mm'].quantile(.99):.3f} mm"
        )
        print(
            f"Máximo por evento: "
            f"{events['irrigation_depth_mm'].max():.3f} mm"
        )
        print(
            f"Eventos no teto de {MAX_IRRIGATION_DEPTH_MM:g} mm: "
            f"{(events['irrigation_depth_mm'] >= MAX_IRRIGATION_DEPTH_MM - 1e-9).sum():,}"
        )

    print(
        f"Água total: "
        f"{df['irrigation_depth_mm'].sum():.2f} mm"
    )

    print()
    print(
        f"Umidade mínima: "
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

    print()
    print("DISTRIBUIÇÃO DOS EVENTOS")

    bins = [0, 4, 5, 6, 7, 8, 10, 12, 15, np.inf]
    labels = [
        "0–4",
        "4–5",
        "5–6",
        "6–7",
        "7–8",
        "8–10",
        "10–12",
        "12–15",
        ">15",
    ]

    distribution = pd.cut(
        events["irrigation_depth_mm"],
        bins=bins,
        labels=labels,
        right=False,
    ).value_counts().sort_index()

    for label, count in distribution.items():
        pct = 100 * count / len(events) if len(events) else 0
        print(f"  {label:>5} mm | {count:6,} | {pct:6.2f}%")

    print()
    print("EVENTOS POR CENÁRIO")
    print(
        events.groupby("scenario_id")
        .size()
        .to_string()
    )

    print()
    print("EVENTOS POR ESTÁGIO")
    print(
        events.groupby("crop_stage")
        .size()
        .to_string()
    )

    print()
    print("Arquivo salvo em:")
    print(output_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
