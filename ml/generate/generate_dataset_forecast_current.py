from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "data" / "irrigation_dataset.csv"
FORECAST_PATH = BASE_DIR / "data" / "forecast_santo_amaro_2024_2025.csv"
OUTPUT_PATH = BASE_DIR / "data" / "irrigation_dataset_forecast_current.csv"


# ============================================================
# FEATURES HISTÓRICAS
# ============================================================

HISTORICAL_FEATURES = [
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
]


# ============================================================
# FEATURES BRUTAS DO FORECAST
# ============================================================

FORECAST_RAW_FEATURES = [
    "forecast_temperature_1d_c",
    "forecast_humidity_1d_percent",
    "forecast_precipitation_1d_mm",
    "forecast_wind_1d_m_s",
    "forecast_radiation_1d_w_m2",
    "forecast_et0_1d_mm",

    "forecast_temperature_2d_c",
    "forecast_humidity_2d_percent",
    "forecast_precipitation_2d_mm",
    "forecast_wind_2d_m_s",
    "forecast_radiation_2d_w_m2",
    "forecast_et0_2d_mm",

    "forecast_temperature_3d_c",
    "forecast_humidity_3d_percent",
    "forecast_precipitation_3d_mm",
    "forecast_wind_3d_m_s",
    "forecast_radiation_3d_w_m2",
    "forecast_et0_3d_mm",
]


# ============================================================
# FEATURES FINAIS DE FORECAST
# ============================================================

FORECAST_FEATURES = [
    "forecast_temperature_24h_c",
    "forecast_humidity_24h_percent",
    "forecast_precipitation_24h_mm",
    "forecast_wind_24h_m_s",
    "forecast_radiation_24h_w_m2",
    "forecast_et0_24h_mm",

    "forecast_temperature_48h_c",
    "forecast_humidity_48h_percent",
    "forecast_precipitation_48h_mm",
    "forecast_wind_48h_m_s",
    "forecast_radiation_48h_w_m2",
    "forecast_et0_48h_mm",

    "forecast_temperature_72h_c",
    "forecast_humidity_72h_percent",
    "forecast_precipitation_72h_mm",
    "forecast_wind_72h_m_s",
    "forecast_radiation_72h_w_m2",
    "forecast_et0_72h_mm",

    "forecast_rain_total_72h",
    "forecast_temperature_mean_72h",
    "forecast_humidity_mean_72h",
    "forecast_wind_mean_72h",
    "forecast_radiation_mean_72h",
    "forecast_et0_mean_72h",
]


# ============================================================
# TARGETS
# ============================================================

TARGETS = [
    "irrigation_depth_current",
    "irrigation_event_current",
]


# ============================================================
# CARREGAR DATASET PRINCIPAL
# ============================================================

def load_base_dataset():

    print("=" * 70)
    print("CARREGANDO DATASET BASE")
    print("=" * 70)

    print(f"Arquivo: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(
        ["scenario_id", "timestamp"]
    ).reset_index(drop=True)

    print(f"Registros: {len(df):,}")
    print(f"Cenários: {df['scenario_id'].nunique()}")
    print(
        f"Período: "
        f"{df['timestamp'].min()} → "
        f"{df['timestamp'].max()}"
    )

    return df


# ============================================================
# FEATURES HISTÓRICAS
# ============================================================

def add_historical_features(df):

    print("\n" + "=" * 70)
    print("GERANDO FEATURES HISTÓRICAS")
    print("=" * 70)

    df = df.sort_values(
        ["scenario_id", "timestamp"]
    ).copy()

    grouped = df.groupby("scenario_id", group_keys=False)

    # --------------------------------------------------------
    # LAGS DE UMIDADE
    # --------------------------------------------------------

    df["soil_moisture_lag_1h"] = grouped[
        "soil_moisture_percent"
    ].shift(1)

    df["soil_moisture_lag_3h"] = grouped[
        "soil_moisture_percent"
    ].shift(3)

    df["soil_moisture_lag_6h"] = grouped[
        "soil_moisture_percent"
    ].shift(6)

    # --------------------------------------------------------
    # ROLLING ETc
    # --------------------------------------------------------

    df["etc_6h"] = (
        grouped["etc"]
        .rolling(6, min_periods=6)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["etc_24h"] = (
        grouped["etc"]
        .rolling(24, min_periods=24)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # --------------------------------------------------------
    # ROLLING CHUVA
    # --------------------------------------------------------

    df["rain_6h"] = (
        grouped["precipitation_mm"]
        .rolling(6, min_periods=6)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["rain_24h"] = (
        grouped["precipitation_mm"]
        .rolling(24, min_periods=24)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # --------------------------------------------------------
    # TEMPERATURA MÉDIA 6H
    # --------------------------------------------------------

    df["temperature_6h_mean"] = (
        grouped["temperature_c"]
        .rolling(6, min_periods=6)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # --------------------------------------------------------
    # RADIAÇÃO ACUMULADA 6H
    # --------------------------------------------------------

    df["radiation_6h_sum"] = (
        grouped["solar_radiation_kwh_m2"]
        .rolling(6, min_periods=6)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # --------------------------------------------------------
    # VARIÁVEIS TEMPORAIS
    # --------------------------------------------------------

    df["hour_sin"] = np.sin(
        2 * np.pi * df["timestamp"].dt.hour / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["timestamp"].dt.hour / 24
    )

    day_of_year = df["timestamp"].dt.dayofyear

    df["day_of_year_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    df["day_of_year_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    print("Features históricas geradas.")

    return df


# ============================================================
# CARREGAR FORECAST
# ============================================================

def load_forecast():

    print("\n" + "=" * 70)
    print("CARREGANDO FORECAST")
    print("=" * 70)

    print(f"Arquivo: {FORECAST_PATH}")

    forecast = pd.read_csv(FORECAST_PATH)

    forecast["timestamp"] = pd.to_datetime(
        forecast["timestamp"]
    )

    print(f"Registros originais: {len(forecast):,}")

    missing_columns = [
        col
        for col in FORECAST_RAW_FEATURES
        if col not in forecast.columns
    ]

    if missing_columns:

        raise RuntimeError(
            "Colunas de forecast ausentes:\n"
            + "\n".join(missing_columns)
        )

    return forecast


# ============================================================
# LIMPEZA DO FORECAST
# ============================================================

def clean_forecast(forecast):

    print("\n" + "=" * 70)
    print("VALIDANDO E LIMPANDO FORECAST")
    print("=" * 70)

    forecast = forecast.copy()

    radiation_columns = [
        "forecast_radiation_1d_w_m2",
        "forecast_radiation_2d_w_m2",
        "forecast_radiation_3d_w_m2",
    ]

    # ========================================================
    # RADIAÇÃO SOLAR
    # ========================================================
    #
    # O valor -1 W/m² não representa radiação física negativa.
    # Nos registros encontrados, ele ocorre às 03:00, durante
    # o período noturno.
    #
    # Portanto, nesses casos, o valor é tratado como 0 W/m².
    #
    # IMPORTANTE:
    # NÃO removemos o timestamp.
    # ========================================================

    for col in radiation_columns:

        negative_mask = forecast[col] < 0

        negative_count = negative_mask.sum()

        print(
            f"{col}: "
            f"{negative_count:,} valores negativos"
        )

        if negative_count > 0:

            negative_hours = (
                forecast.loc[
                    negative_mask,
                    "timestamp"
                ]
                .dt.hour
                .value_counts()
                .sort_index()
            )

            print(
                "Horários dos valores negativos:"
            )

            print(negative_hours)

            # Verificação de segurança:
            # somente valores noturnos podem ser convertidos
            # automaticamente para zero.
            invalid_daytime = (
                negative_mask
                & forecast["timestamp"].dt.hour.between(
                    6, 18
                )
            )

            if invalid_daytime.any():

                problematic = forecast.loc[
                    invalid_daytime,
                    [
                        "timestamp",
                        col,
                    ]
                ]

                raise RuntimeError(
                    "Foi encontrada radiação negativa "
                    "durante período diurno:\n"
                    f"{problematic.head(20)}"
                )

            # Corrige somente o valor inválido.
            forecast.loc[
                negative_mask,
                col
            ] = 0.0

            print(
                f"  {negative_count:,} valores "
                f"corrigidos para 0.0 W/m²."
            )

    # ========================================================
    # OUTRAS VERIFICAÇÕES
    # ========================================================

    temperature_columns = [
        col
        for col in forecast.columns
        if col.startswith("forecast_temperature_")
    ]

    humidity_columns = [
        col
        for col in forecast.columns
        if col.startswith("forecast_humidity_")
    ]

    precipitation_columns = [
        col
        for col in forecast.columns
        if col.startswith("forecast_precipitation_")
    ]

    wind_columns = [
        col
        for col in forecast.columns
        if col.startswith("forecast_wind_")
    ]

    et0_columns = [
        col
        for col in forecast.columns
        if col.startswith("forecast_et0_")
    ]

    # --------------------------------------------------------
    # Temperatura
    # --------------------------------------------------------

    for col in temperature_columns:

        if (
            forecast[col].min() < -10
            or forecast[col].max() > 60
        ):

            raise RuntimeError(
                f"Temperatura fora da faixa plausível: {col}"
            )

    # --------------------------------------------------------
    # Umidade
    # --------------------------------------------------------

    for col in humidity_columns:

        if (
            forecast[col].min() < 0
            or forecast[col].max() > 100
        ):

            raise RuntimeError(
                f"Umidade fora da faixa plausível: {col}"
            )

    # --------------------------------------------------------
    # Precipitação
    # --------------------------------------------------------

    for col in precipitation_columns:

        if forecast[col].min() < 0:

            raise RuntimeError(
                f"Precipitação negativa encontrada: {col}"
            )

    # --------------------------------------------------------
    # Vento
    # --------------------------------------------------------

    for col in wind_columns:

        if forecast[col].min() < 0:

            raise RuntimeError(
                f"Vento negativo encontrado: {col}"
            )

    # --------------------------------------------------------
    # Radiação
    # --------------------------------------------------------

    for col in radiation_columns:

        if forecast[col].min() < 0:

            raise RuntimeError(
                f"Radiação negativa restante: {col}"
            )

    # --------------------------------------------------------
    # ET0
    # --------------------------------------------------------

    for col in et0_columns:

        if forecast[col].min() < 0:

            raise RuntimeError(
                f"ET0 negativa encontrada: {col}"
            )

    print(
        "\nValidação das faixas meteorológicas: OK"
    )

    return forecast

# ============================================================
# PREPARAR FEATURES DO FORECAST
# ============================================================

def prepare_forecast_features(forecast):

    print("\n" + "=" * 70)
    print("PREPARANDO FEATURES DE FORECAST")
    print("=" * 70)

    forecast = forecast[
        ["timestamp"] + FORECAST_RAW_FEATURES
    ].copy()

    forecast = forecast.rename(
        columns={

            "forecast_temperature_1d_c":
                "forecast_temperature_24h_c",

            "forecast_humidity_1d_percent":
                "forecast_humidity_24h_percent",

            "forecast_precipitation_1d_mm":
                "forecast_precipitation_24h_mm",

            "forecast_wind_1d_m_s":
                "forecast_wind_24h_m_s",

            "forecast_radiation_1d_w_m2":
                "forecast_radiation_24h_w_m2",

            "forecast_et0_1d_mm":
                "forecast_et0_24h_mm",


            "forecast_temperature_2d_c":
                "forecast_temperature_48h_c",

            "forecast_humidity_2d_percent":
                "forecast_humidity_48h_percent",

            "forecast_precipitation_2d_mm":
                "forecast_precipitation_48h_mm",

            "forecast_wind_2d_m_s":
                "forecast_wind_48h_m_s",

            "forecast_radiation_2d_w_m2":
                "forecast_radiation_48h_w_m2",

            "forecast_et0_2d_mm":
                "forecast_et0_48h_mm",


            "forecast_temperature_3d_c":
                "forecast_temperature_72h_c",

            "forecast_humidity_3d_percent":
                "forecast_humidity_72h_percent",

            "forecast_precipitation_3d_mm":
                "forecast_precipitation_72h_mm",

            "forecast_wind_3d_m_s":
                "forecast_wind_72h_m_s",

            "forecast_radiation_3d_w_m2":
                "forecast_radiation_72h_w_m2",

            "forecast_et0_3d_mm":
                "forecast_et0_72h_mm",
        }
    )

    return forecast


# ============================================================
# ALINHAR FORECAST AO MOMENTO DA DECISÃO
# ============================================================

def align_forecast_to_decision_time(forecast):

    print("\n" + "=" * 70)
    print("ALINHANDO FORECAST AO MOMENTO DA DECISÃO")
    print("=" * 70)

    forecast = forecast.copy()

    # --------------------------------------------------------
    # +24H
    #
    # previous_day1 em T+24 representa a previsão
    # disponível aproximadamente em T.
    # --------------------------------------------------------

    f24 = forecast[
        [
            "timestamp",
            "forecast_temperature_24h_c",
            "forecast_humidity_24h_percent",
            "forecast_precipitation_24h_mm",
            "forecast_wind_24h_m_s",
            "forecast_radiation_24h_w_m2",
            "forecast_et0_24h_mm",
        ]
    ].copy()

    f24["timestamp"] = (
        f24["timestamp"]
        - pd.Timedelta(hours=24)
    )

    # --------------------------------------------------------
    # +48H
    # --------------------------------------------------------

    f48 = forecast[
        [
            "timestamp",
            "forecast_temperature_48h_c",
            "forecast_humidity_48h_percent",
            "forecast_precipitation_48h_mm",
            "forecast_wind_48h_m_s",
            "forecast_radiation_48h_w_m2",
            "forecast_et0_48h_mm",
        ]
    ].copy()

    f48["timestamp"] = (
        f48["timestamp"]
        - pd.Timedelta(hours=48)
    )

    # --------------------------------------------------------
    # +72H
    # --------------------------------------------------------

    f72 = forecast[
        [
            "timestamp",
            "forecast_temperature_72h_c",
            "forecast_humidity_72h_percent",
            "forecast_precipitation_72h_mm",
            "forecast_wind_72h_m_s",
            "forecast_radiation_72h_w_m2",
            "forecast_et0_72h_mm",
        ]
    ].copy()

    f72["timestamp"] = (
        f72["timestamp"]
        - pd.Timedelta(hours=72)
    )

    # --------------------------------------------------------
    # MERGES
    # --------------------------------------------------------

    forecast_aligned = f24.merge(
        f48,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )

    forecast_aligned = forecast_aligned.merge(
        f72,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Registros após alinhamento: "
        f"{len(forecast_aligned):,}"
    )

    return forecast_aligned


# ============================================================
# AGREGAÇÕES DO FORECAST
# ============================================================

def add_forecast_aggregates(forecast):

    print("\n" + "=" * 70)
    print("GERANDO AGREGAÇÕES DO FORECAST")
    print("=" * 70)

    forecast = forecast.copy()

    # --------------------------------------------------------
    # CHUVA TOTAL 72H
    # --------------------------------------------------------

    forecast["forecast_rain_total_72h"] = (
        forecast["forecast_precipitation_24h_mm"]
        + forecast["forecast_precipitation_48h_mm"]
        + forecast["forecast_precipitation_72h_mm"]
    )

    # --------------------------------------------------------
    # MÉDIAS 72H
    # --------------------------------------------------------

    forecast["forecast_temperature_mean_72h"] = (
        forecast[
            [
                "forecast_temperature_24h_c",
                "forecast_temperature_48h_c",
                "forecast_temperature_72h_c",
            ]
        ].mean(axis=1)
    )

    forecast["forecast_humidity_mean_72h"] = (
        forecast[
            [
                "forecast_humidity_24h_percent",
                "forecast_humidity_48h_percent",
                "forecast_humidity_72h_percent",
            ]
        ].mean(axis=1)
    )

    forecast["forecast_wind_mean_72h"] = (
        forecast[
            [
                "forecast_wind_24h_m_s",
                "forecast_wind_48h_m_s",
                "forecast_wind_72h_m_s",
            ]
        ].mean(axis=1)
    )

    forecast["forecast_radiation_mean_72h"] = (
        forecast[
            [
                "forecast_radiation_24h_w_m2",
                "forecast_radiation_48h_w_m2",
                "forecast_radiation_72h_w_m2",
            ]
        ].mean(axis=1)
    )

    forecast["forecast_et0_mean_72h"] = (
        forecast[
            [
                "forecast_et0_24h_mm",
                "forecast_et0_48h_mm",
                "forecast_et0_72h_mm",
            ]
        ].mean(axis=1)
    )

    return forecast


# ============================================================
# TARGET ATUAL
# ============================================================

def add_current_target(df):

    print("\n" + "=" * 70)
    print("GERANDO TARGET ATUAL")
    print("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # IMPORTANTÍSSIMO:
    #
    # irrigation_depth_mm representa a decisão de irrigação
    # no próprio timestamp.
    #
    # O solo usado como feature é o solo PRÉ-IRRIGAÇÃO,
    # conforme Dataset 3 / V4.
    # --------------------------------------------------------

    df["irrigation_depth_current"] = (
        df["irrigation_depth_mm"]
    )

    df["irrigation_event_current"] = (
        df["irrigation_depth_current"] > 0
    ).astype(int)

    print(
        f"Eventos atuais: "
        f"{df['irrigation_event_current'].sum():,}"
    )

    print(
        f"Não eventos: "
        f"{(df['irrigation_event_current'] == 0).sum():,}"
    )

    print(
        f"Taxa de eventos: "
        f"{df['irrigation_event_current'].mean() * 100:.4f}%"
    )

    return df


# ============================================================
# VALIDAR DATASET FINAL
# ============================================================

def validate_final_dataset(df):

    print("\n" + "=" * 70)
    print("VALIDANDO DATASET FINAL")
    print("=" * 70)

    expected_features = (
        HISTORICAL_FEATURES
        + FORECAST_FEATURES
    )

    expected_columns = [
        "timestamp",
        "scenario_id",
        *expected_features,
        *TARGETS,
    ]

    missing = [
        col
        for col in expected_columns
        if col not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Colunas esperadas ausentes:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # DUPLICIDADES
    # --------------------------------------------------------

    duplicate_key = df.duplicated(
        subset=[
            "timestamp",
            "scenario_id",
        ]
    ).sum()

    if duplicate_key > 0:

        raise RuntimeError(
            f"Foram encontradas "
            f"{duplicate_key:,} duplicidades "
            f"em (timestamp, scenario_id)."
        )

    # --------------------------------------------------------
    # NaN
    # --------------------------------------------------------

    nan_total = df[
        expected_features + TARGETS
    ].isna().sum().sum()

    if nan_total > 0:

        raise RuntimeError(
            f"Foram encontrados "
            f"{nan_total:,} valores NaN "
            f"nas features/targets."
        )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    inconsistent_target = (
        (
            df["irrigation_depth_current"] > 0
        ).astype(int)
        != df["irrigation_event_current"]
    ).sum()

    if inconsistent_target > 0:

        raise RuntimeError(
            "Inconsistência entre "
            "irrigation_depth_current e "
            "irrigation_event_current."
        )

    # --------------------------------------------------------
    # RADIAÇÃO
    # --------------------------------------------------------

    radiation_cols = [
        col
        for col in FORECAST_FEATURES
        if "radiation" in col
    ]

    negative_radiation = (
        df[radiation_cols] < 0
    ).sum().sum()

    if negative_radiation > 0:

        raise RuntimeError(
            "Ainda existem valores negativos "
            "de radiação no dataset final."
        )

    # --------------------------------------------------------
    # CENÁRIOS
    # --------------------------------------------------------

    timestamps = df["timestamp"].nunique()
    scenarios = df["scenario_id"].nunique()

    expected_records = timestamps * scenarios

    if len(df) != expected_records:

        raise RuntimeError(
            "Incompletude dos cenários:\n"
            f"Esperados: {expected_records:,}\n"
            f"Encontrados: {len(df):,}"
        )

    print("Chave (timestamp, scenario_id): OK")
    print("NaN: OK")
    print("Target: OK")
    print("Radiação: OK")
    print("Cenários: OK")

    print(
        f"\nFeatures finais: "
        f"{len(expected_features)}"
    )

    print(
        f"Targets finais: "
        f"{len(TARGETS)}"
    )

    print(
        f"Total de colunas finais: "
        f"{len(expected_columns)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GERAÇÃO DO DATASET FORECAST-CURRENT")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. DATASET BASE
    # --------------------------------------------------------

    df = load_base_dataset()

    # --------------------------------------------------------
    # 2. FEATURES HISTÓRICAS
    # --------------------------------------------------------

    df = add_historical_features(df)

    # --------------------------------------------------------
    # 3. TARGET ATUAL
    # --------------------------------------------------------

    df = add_current_target(df)

    # --------------------------------------------------------
    # 4. FORECAST
    # --------------------------------------------------------

    forecast = load_forecast()

    # --------------------------------------------------------
    # 5. LIMPEZA
    # --------------------------------------------------------

    forecast = clean_forecast(forecast)

    # --------------------------------------------------------
    # 6. PREPARAR FEATURES
    # --------------------------------------------------------

    forecast = prepare_forecast_features(
        forecast
    )

    # --------------------------------------------------------
    # 7. ALINHAR +24/+48/+72
    # --------------------------------------------------------

    forecast = align_forecast_to_decision_time(
        forecast
    )

    # --------------------------------------------------------
    # 8. AGREGAÇÕES
    # --------------------------------------------------------

    forecast = add_forecast_aggregates(
        forecast
    )

    # --------------------------------------------------------
    # 9. MERGE COM DATASET BASE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INTEGRANDO DATASET BASE + FORECAST")
    print("=" * 70)

    df = df.merge(
        forecast,
        on="timestamp",
        how="inner",
        validate="many_to_one",
    )

    print(
        f"Registros após merge: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # 10. REMOVER LINHAS SEM FEATURES HISTÓRICAS
    # --------------------------------------------------------

    required_columns = (
        HISTORICAL_FEATURES
        + FORECAST_FEATURES
        + TARGETS
    )

    before = len(df)

    df = df.dropna(
        subset=required_columns
    ).copy()

    removed = before - len(df)

    print(
        f"Registros removidos por NaN: "
        f"{removed:,}"
    )

    # --------------------------------------------------------
    # 11. SELECIONAR COLUNAS FINAIS
    # --------------------------------------------------------

    final_columns = [
        "timestamp",
        "scenario_id",
        *HISTORICAL_FEATURES,
        *FORECAST_FEATURES,
        *TARGETS,
    ]

    df = df[final_columns].copy()

    # --------------------------------------------------------
    # 12. ORDENAR
    # --------------------------------------------------------

    df = df.sort_values(
        ["timestamp", "scenario_id"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 13. VALIDAÇÃO FINAL
    # --------------------------------------------------------

    validate_final_dataset(df)

    # --------------------------------------------------------
    # 14. SALVAR
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SALVANDO DATASET")
    print("=" * 70)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Arquivo salvo em:\n"
        f"{OUTPUT_PATH}"
    )

    # --------------------------------------------------------
    # 15. RESUMO
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESUMO FINAL")
    print("=" * 70)

    print(
        f"Registros: {len(df):,}"
    )

    print(
        f"Colunas: {len(df.columns)}"
    )

    print(
        f"Features históricas: "
        f"{len(HISTORICAL_FEATURES)}"
    )

    print(
        f"Features de forecast: "
        f"{len(FORECAST_FEATURES)}"
    )

    print(
        f"Targets: {len(TARGETS)}"
    )

    print(
        f"Cenários: "
        f"{df['scenario_id'].nunique()}"
    )

    print(
        f"Eventos atuais: "
        f"{df['irrigation_event_current'].sum():,}"
    )

    print(
        f"Taxa de eventos: "
        f"{df['irrigation_event_current'].mean() * 100:.4f}%"
    )

    print("\nDATASET FORECAST-CURRENT GERADO COM SUCESSO.")


if __name__ == "__main__":
    main()