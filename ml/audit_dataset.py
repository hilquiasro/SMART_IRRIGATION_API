from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "irrigation_dataset_2.csv"
RESULTS_DIR = BASE_DIR / "dataset_audit"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


TARGET = "irrigation_depth_mm"


# ============================================================
# CARREGAMENTO
# ============================================================

print("\nCarregando dataset...")

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Registros: {len(df):,}")
print(f"Colunas:   {len(df.columns)}")


# ============================================================
# 1. ESTRUTURA
# ============================================================

print("\n" + "=" * 70)
print("1. ESTRUTURA DO DATASET")
print("=" * 70)

print("\nTipos:")
print(df.dtypes)

print("\nValores ausentes:")
print(df.isna().sum())

print("\nDuplicados:")
print(df.duplicated().sum())


# ============================================================
# 2. DISTRIBUIÇÃO DO TARGET
# ============================================================

print("\n" + "=" * 70)
print("2. DISTRIBUIÇÃO DO TARGET")
print("=" * 70)

target = df[TARGET]

print(target.describe())

zero_count = (target == 0).sum()
event_count = (target > 0).sum()

print(f"\nSem irrigação: {zero_count:,}")
print(f"Com irrigação: {event_count:,}")

print(
    f"Percentual de eventos: "
    f"{event_count / len(df) * 100:.3f}%"
)

print("\nQuantis do target:")
print(
    target.quantile(
        [
            0,
            0.5,
            0.75,
            0.90,
            0.95,
            0.99,
            1.0,
        ]
    )
)


# ============================================================
# 3. DISTRIBUIÇÃO DOS EVENTOS
# ============================================================

events = df[df[TARGET] > 0].copy()

print("\n" + "=" * 70)
print("3. DISTRIBUIÇÃO DAS IRRIGAÇÕES")
print("=" * 70)

bins = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    np.inf,
]

labels = [
    "0–1",
    "1–2",
    "2–3",
    "3–4",
    "4–5",
    "5–6",
    "6–7",
    "7–8",
    ">8",
]

event_distribution = pd.cut(
    events[TARGET],
    bins=bins,
    labels=labels,
    right=False,
)

distribution = (
    event_distribution
    .value_counts()
    .sort_index()
)

distribution_percent = (
    distribution
    / distribution.sum()
    * 100
)

event_distribution_table = pd.DataFrame({
    "events": distribution,
    "percentage": distribution_percent,
})

print(event_distribution_table)


# ============================================================
# 4. IRRIGAÇÃO POR ESTÁGIO
# ============================================================

print("\n" + "=" * 70)
print("4. IRRIGAÇÃO POR ESTÁGIO")
print("=" * 70)

stage_analysis = (
    df.groupby("crop_stage", observed=False)
    .agg(
        records=(TARGET, "size"),
        events=(TARGET, lambda x: (x > 0).sum()),
        total_irrigation_mm=(TARGET, "sum"),
        mean_irrigation_mm=(TARGET, "mean"),
        mean_event_mm=(
            TARGET,
            lambda x: x[x > 0].mean()
            if (x > 0).any()
            else 0,
        ),
    )
)

stage_analysis["event_rate_percent"] = (
    stage_analysis["events"]
    / stage_analysis["records"]
    * 100
)

print(stage_analysis)


# ============================================================
# 5. IRRIGAÇÃO POR FAIXA DE UMIDADE
# ============================================================

print("\n" + "=" * 70)
print("5. IRRIGAÇÃO × UMIDADE DO SOLO")
print("=" * 70)

moisture_bins = [
    0,
    40,
    42,
    44,
    46,
    48,
    50,
    52,
    54,
    56,
    58,
    60,
    65,
    70,
    80,
    100,
    np.inf,
]

moisture_labels = [
    "<40",
    "40–42",
    "42–44",
    "44–46",
    "46–48",
    "48–50",
    "50–52",
    "52–54",
    "54–56",
    "56–58",
    "58–60",
    "60–65",
    "65–70",
    "70–80",
    "80–100",
    ">100",
]

df["moisture_range"] = pd.cut(
    df["soil_moisture_percent"],
    bins=moisture_bins,
    labels=moisture_labels,
    right=False,
)

moisture_analysis = (
    df.groupby(
        "moisture_range",
        observed=False,
    )
    .agg(
        records=(TARGET, "size"),
        events=(TARGET, lambda x: (x > 0).sum()),
        mean_irrigation_mm=(TARGET, "mean"),
        mean_event_mm=(
            TARGET,
            lambda x: x[x > 0].mean()
            if (x > 0).any()
            else 0,
        ),
    )
)

moisture_analysis["event_rate_percent"] = (
    moisture_analysis["events"]
    / moisture_analysis["records"]
    * 100
)

print(moisture_analysis)


# ============================================================
# 6. IRRIGAÇÃO × VARIÁVEIS METEOROLÓGICAS
# ============================================================

print("\n" + "=" * 70)
print("6. CORRELAÇÕES COM O TARGET")
print("=" * 70)

numeric_columns = [
    "temperature_c",
    "air_humidity_percent",
    "wind_speed_m_s",
    "solar_radiation_kwh_m2",
    "precipitation_mm",
    "soil_moisture_percent",
    "kc",
    "et0",
    "etc",
    "dynamic_lower_bound_percent",
    "hour_sin",
    "hour_cos",
    TARGET,
]

correlations = (
    df[numeric_columns]
    .corr(numeric_only=True)[TARGET]
    .sort_values()
)

print(correlations)


# ============================================================
# 7. ESTATÍSTICAS DOS EVENTOS
# ============================================================

print("\n" + "=" * 70)
print("7. ESTATÍSTICAS DOS EVENTOS")
print("=" * 70)

event_stats = events[TARGET].describe()

print(event_stats)


# ============================================================
# 8. IRRIGAÇÃO × CHUVA
# ============================================================

print("\n" + "=" * 70)
print("8. IRRIGAÇÃO × PRECIPITAÇÃO")
print("=" * 70)

rain_bins = [
    -0.001,
    0,
    0.1,
    1,
    2,
    4,
    6,
    10,
    np.inf,
]

rain_labels = [
    "0",
    "0–0.1",
    "0.1–1",
    "1–2",
    "2–4",
    "4–6",
    "6–10",
    ">10",
]

df["rain_range"] = pd.cut(
    df["precipitation_mm"],
    bins=rain_bins,
    labels=rain_labels,
    right=False,
)

rain_analysis = (
    df.groupby(
        "rain_range",
        observed=False,
    )
    .agg(
        records=(TARGET, "size"),
        events=(TARGET, lambda x: (x > 0).sum()),
        mean_irrigation_mm=(TARGET, "mean"),
    )
)

rain_analysis["event_rate_percent"] = (
    rain_analysis["events"]
    / rain_analysis["records"]
    * 100
)

print(rain_analysis)


# ============================================================
# 9. IRRIGAÇÃO × ETC
# ============================================================

print("\n" + "=" * 70)
print("9. IRRIGAÇÃO × ETC")
print("=" * 70)

etc_bins = [
    -0.001,
    0,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    np.inf,
]

etc_labels = [
    "0",
    "0–0.05",
    "0.05–0.10",
    "0.10–0.20",
    "0.20–0.30",
    "0.30–0.40",
    "0.40–0.50",
    "0.50–0.60",
    "0.60–0.70",
    ">0.70",
]

df["etc_range"] = pd.cut(
    df["etc"],
    bins=etc_bins,
    labels=etc_labels,
    right=False,
)

etc_analysis = (
    df.groupby(
        "etc_range",
        observed=False,
    )
    .agg(
        records=(TARGET, "size"),
        events=(TARGET, lambda x: (x > 0).sum()),
        mean_irrigation_mm=(TARGET, "mean"),
    )
)

etc_analysis["event_rate_percent"] = (
    etc_analysis["events"]
    / etc_analysis["records"]
    * 100
)

print(etc_analysis)


# ============================================================
# 10. AUTOCORRELAÇÃO TEMPORAL DO TARGET
# ============================================================

print("\n" + "=" * 70)
print("10. AUTOCORRELAÇÃO TEMPORAL")
print("=" * 70)

target_lag_1 = target.shift(1)

autocorrelation = target.corr(target_lag_1)

print(
    f"Correlação target(t) × target(t-1): "
    f"{autocorrelation:.6f}"
)


# ============================================================
# 11. VARIAÇÃO DO TARGET EM CONDIÇÕES SEMELHANTES
# ============================================================

print("\n" + "=" * 70)
print("11. VARIAÇÃO DO TARGET EM CONDIÇÕES SEMELHANTES")
print("=" * 70)

comparison_columns = [
    "soil_moisture_percent",
    "temperature_c",
    "air_humidity_percent",
    "solar_radiation_kwh_m2",
    "crop_stage",
    "hour_sin",
    "hour_cos",
]

# Arredondamos as variáveis contínuas para identificar
# situações aproximadamente semelhantes.
similar_df = df.copy()

similar_df["soil_bin"] = (
    similar_df["soil_moisture_percent"]
    .round()
)

similar_df["temp_bin"] = (
    similar_df["temperature_c"]
    .round(1)
)

similar_df["humidity_bin"] = (
    similar_df["air_humidity_percent"]
    .round()
)

similar_df["solar_bin"] = (
    similar_df["solar_radiation_kwh_m2"]
    .round(1)
)

similar_df["hour_sin_bin"] = (
    similar_df["hour_sin"]
    .round(1)
)

similar_df["hour_cos_bin"] = (
    similar_df["hour_cos"]
    .round(1)
)

group_columns = [
    "soil_bin",
    "temp_bin",
    "humidity_bin",
    "solar_bin",
    "crop_stage",
    "hour_sin_bin",
    "hour_cos_bin",
]

group_variability = (
    similar_df
    .groupby(group_columns, observed=False)
    .agg(
        records=(TARGET, "size"),
        target_mean=(TARGET, "mean"),
        target_std=(TARGET, "std"),
        target_min=(TARGET, "min"),
        target_max=(TARGET, "max"),
        events=(
            TARGET,
            lambda x: (x > 0).sum(),
        ),
    )
)

group_variability = group_variability[
    group_variability["records"] >= 5
]

group_variability["event_rate"] = (
    group_variability["events"]
    / group_variability["records"]
)

print(
    "\nGrupos semelhantes encontrados:",
    len(group_variability),
)

print(
    "\nMaior variabilidade do target:"
)

print(
    group_variability
    .sort_values(
        "target_std",
        ascending=False,
    )
    .head(20)
)


# ============================================================
# 12. SALVAR RESULTADOS
# ============================================================

event_distribution_table.to_csv(
    RESULTS_DIR / "event_distribution.csv"
)

stage_analysis.to_csv(
    RESULTS_DIR / "irrigation_by_stage.csv"
)

moisture_analysis.to_csv(
    RESULTS_DIR / "irrigation_by_moisture.csv"
)

rain_analysis.to_csv(
    RESULTS_DIR / "irrigation_by_rain.csv"
)

etc_analysis.to_csv(
    RESULTS_DIR / "irrigation_by_etc.csv"
)

correlations.to_csv(
    RESULTS_DIR / "target_correlations.csv"
)

group_variability.to_csv(
    RESULTS_DIR / "similar_condition_variability.csv"
)


# ============================================================
# RESUMO
# ============================================================

summary = pd.DataFrame({
    "metric": [
        "records",
        "zero_records",
        "irrigation_events",
        "event_rate_percent",
        "target_mean",
        "event_mean",
        "event_std",
        "event_min",
        "event_max",
        "target_autocorrelation_lag1",
    ],
    "value": [
        len(df),
        zero_count,
        event_count,
        event_count / len(df) * 100,
        target.mean(),
        events[TARGET].mean(),
        events[TARGET].std(),
        events[TARGET].min(),
        events[TARGET].max(),
        autocorrelation,
    ],
})

summary.to_csv(
    RESULTS_DIR / "audit_summary.csv",
    index=False,
)

print("\n" + "=" * 70)
print("AUDITORIA CONCLUÍDA")
print("=" * 70)

print(
    f"Resultados salvos em: {RESULTS_DIR}"
)