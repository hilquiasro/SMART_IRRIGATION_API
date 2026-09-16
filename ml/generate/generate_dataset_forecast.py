from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATASET_3_PATH = DATA_DIR / "irrigation_dataset.csv"
NASA_PATH = DATA_DIR / "weather_santo_amaro_2020_2025_hourly.csv"
FORECAST_PATH = DATA_DIR / "forecast_santo_amaro_2024_2025.csv"

OUTPUT_PATH = DATA_DIR / "irrigation_dataset_forecast.csv"

LATITUDE = -12.5467
LONGITUDE = -38.7119

RANDOM_SEED = 42


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def check_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{path}"
        )


def add_historical_features(df):
    """
    Cria as variáveis históricas usadas pelo modelo.
    """

    df = df.sort_values(
        ["scenario_id", "timestamp"]
    ).copy()

    grouped = df.groupby("scenario_id", sort=False)

    # --------------------------------------------------------
    # Defasagens da umidade do solo
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ETc acumulada
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
    # Chuva acumulada
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
    # Temperatura média
    # --------------------------------------------------------

    df["temperature_6h_mean"] = (
        grouped["temperature_c"]
        .rolling(6, min_periods=6)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # --------------------------------------------------------
    # Radiação acumulada
    # --------------------------------------------------------

    df["radiation_6h_sum"] = (
        grouped["solar_radiation_kwh_m2"]
        .rolling(6, min_periods=6)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # --------------------------------------------------------
    # Variáveis cíclicas
    # --------------------------------------------------------

    hour = df["timestamp"].dt.hour

    df["hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    day_of_year = df["timestamp"].dt.dayofyear

    df["day_of_year_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    df["day_of_year_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    return df


# ============================================================
# CARREGAMENTO DA PREVISÃO
# ============================================================

def load_forecast():
    print("\nCarregando previsões meteorológicas...")

    check_file(FORECAST_PATH)

    forecast = pd.read_csv(
        FORECAST_PATH
    )

    forecast["timestamp"] = pd.to_datetime(
        forecast["timestamp"]
    )

    forecast = forecast.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Nomes finais das variáveis
    # --------------------------------------------------------

    rename_map = {
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

    forecast = forecast.rename(
        columns=rename_map
    )

    return forecast


# ============================================================
# ALINHAMENTO DAS PREVISÕES
# ============================================================

def align_forecasts(forecast):
    """
    O arquivo do Previous Runs contém previsões associadas
    ao timestamp previsto.

    Para cada horizonte:

        previous_day1 -> previsão de +24h
        previous_day2 -> previsão de +48h
        previous_day3 -> previsão de +72h

    O timestamp é deslocado para representar o momento
    em que aquela previsão seria utilizada.
    """

    forecast = forecast.copy()

    base_columns = [
        "timestamp"
    ]

    # --------------------------------------------------------
    # +24h
    # --------------------------------------------------------

    columns_24 = [
        "timestamp",
        "forecast_temperature_24h_c",
        "forecast_humidity_24h_percent",
        "forecast_precipitation_24h_mm",
        "forecast_wind_24h_m_s",
        "forecast_radiation_24h_w_m2",
        "forecast_et0_24h_mm",
    ]

    columns_24 = [
        c for c in columns_24
        if c in forecast.columns
    ]

    f24 = forecast[columns_24].copy()

    f24["timestamp"] = (
        f24["timestamp"]
        - pd.Timedelta(hours=24)
    )

    # --------------------------------------------------------
    # +48h
    # --------------------------------------------------------

    columns_48 = [
        "timestamp",
        "forecast_temperature_48h_c",
        "forecast_humidity_48h_percent",
        "forecast_precipitation_48h_mm",
        "forecast_wind_48h_m_s",
        "forecast_radiation_48h_w_m2",
        "forecast_et0_48h_mm",
    ]

    columns_48 = [
        c for c in columns_48
        if c in forecast.columns
    ]

    f48 = forecast[columns_48].copy()

    f48["timestamp"] = (
        f48["timestamp"]
        - pd.Timedelta(hours=48)
    )

    # --------------------------------------------------------
    # +72h
    # --------------------------------------------------------

    columns_72 = [
        "timestamp",
        "forecast_temperature_72h_c",
        "forecast_humidity_72h_percent",
        "forecast_precipitation_72h_mm",
        "forecast_wind_72h_m_s",
        "forecast_radiation_72h_w_m2",
        "forecast_et0_72h_mm",
    ]

    columns_72 = [
        c for c in columns_72
        if c in forecast.columns
    ]

    f72 = forecast[columns_72].copy()

    f72["timestamp"] = (
        f72["timestamp"]
        - pd.Timedelta(hours=72)
    )

    # --------------------------------------------------------
    # Remove duplicatas
    # --------------------------------------------------------

    f24 = f24.drop_duplicates(
        "timestamp"
    )

    f48 = f48.drop_duplicates(
        "timestamp"
    )

    f72 = f72.drop_duplicates(
        "timestamp"
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    aligned = f24.merge(
        f48,
        on="timestamp",
        how="outer"
    )

    aligned = aligned.merge(
        f72,
        on="timestamp",
        how="outer"
    )

    aligned = aligned.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return aligned


# ============================================================
# VARIÁVEIS AGREGADAS DE PREVISÃO
# ============================================================

def add_forecast_features(df):
    """
    Cria variáveis agregadas utilizando as previsões
    de +24h, +48h e +72h.
    """

    # --------------------------------------------------------
    # Chuva prevista total
    # --------------------------------------------------------

    rain_columns = [
        "forecast_precipitation_24h_mm",
        "forecast_precipitation_48h_mm",
        "forecast_precipitation_72h_mm",
    ]

    df["forecast_rain_total_72h"] = (
        df[rain_columns]
        .sum(axis=1, min_count=3)
    )

    # --------------------------------------------------------
    # Temperatura média prevista
    # --------------------------------------------------------

    temp_columns = [
        "forecast_temperature_24h_c",
        "forecast_temperature_48h_c",
        "forecast_temperature_72h_c",
    ]

    df["forecast_temperature_mean_72h"] = (
        df[temp_columns]
        .mean(axis=1)
    )

    # --------------------------------------------------------
    # Umidade média prevista
    # --------------------------------------------------------

    humidity_columns = [
        "forecast_humidity_24h_percent",
        "forecast_humidity_48h_percent",
        "forecast_humidity_72h_percent",
    ]

    df["forecast_humidity_mean_72h"] = (
        df[humidity_columns]
        .mean(axis=1)
    )

    # --------------------------------------------------------
    # Vento médio previsto
    # --------------------------------------------------------

    wind_columns = [
        "forecast_wind_24h_m_s",
        "forecast_wind_48h_m_s",
        "forecast_wind_72h_m_s",
    ]

    df["forecast_wind_mean_72h"] = (
        df[wind_columns]
        .mean(axis=1)
    )

    # --------------------------------------------------------
    # Radiação média prevista
    # --------------------------------------------------------

    radiation_columns = [
        "forecast_radiation_24h_w_m2",
        "forecast_radiation_48h_w_m2",
        "forecast_radiation_72h_w_m2",
    ]

    df["forecast_radiation_mean_72h"] = (
        df[radiation_columns]
        .mean(axis=1)
    )

    # --------------------------------------------------------
    # ET0 média prevista
    # --------------------------------------------------------

    et0_columns = [
        "forecast_et0_24h_mm",
        "forecast_et0_48h_mm",
        "forecast_et0_72h_mm",
    ]

    df["forecast_et0_mean_72h"] = (
        df[et0_columns]
        .mean(axis=1)
    )

    return df


# ============================================================
# TARGET FUTURO
# ============================================================

def future_sum_24h(group):
    """
    Calcula a lâmina de irrigação acumulada nas próximas
    24 horas.

    IMPORTANTE:

    O cálculo só é considerado válido quando existem
    exatamente 24 registros horários consecutivos após
    o timestamp atual.

    Exemplo válido:

        10:00
        11:00
        ...
        33:00

    Exemplo inválido:

        10:00
        11:00
        13:00

    Nesse segundo caso existe um buraco de 2 horas e a
    janela não é utilizada.

    Isso evita que o modelo receba um alvo construído
    artificialmente atravessando uma falha temporal.
    """

    group = group.sort_values(
        "timestamp"
    ).copy()

    timestamps = (
        group["timestamp"]
        .reset_index(drop=True)
    )

    values = (
        group["irrigation_depth_mm"]
        .reset_index(drop=True)
    )

    result = np.full(
        len(group),
        np.nan,
        dtype=float
    )

    timestamp_to_position = {
        timestamp: i
        for i, timestamp in enumerate(timestamps)
    }

    for i, timestamp in enumerate(timestamps):

        target_end = (
            timestamp
            + pd.Timedelta(hours=24)
        )

        end_position = timestamp_to_position.get(
            target_end
        )

        # ----------------------------------------------------
        # Se não existe exatamente t + 24h,
        # a janela não é válida.
        # ----------------------------------------------------

        if end_position is None:
            continue

        # Deve haver exatamente 24 posições entre
        # t e t+24h.

        if end_position != i + 24:
            continue

        future_values = values.iloc[
            i + 1 : i + 25
        ]

        if len(future_values) != 24:
            continue

        result[i] = future_values.sum()

    return pd.Series(
        result,
        index=group.index
    )


# ============================================================
# CRIAÇÃO DO TARGET
# ============================================================

def create_targets(df):
    """
    Cria:

        irrigation_depth_next_24h
        irrigation_event_next_24h

    respeitando a continuidade temporal.
    """

    print("\nCriando alvo futuro de 24h...")

    parts = []

    for scenario_id, group in df.groupby(
        "scenario_id",
        sort=False
    ):

        group = group.sort_values(
            "timestamp"
        ).copy()

        target = future_sum_24h(
            group
        )

        group[
            "irrigation_depth_next_24h"
        ] = target.to_numpy()

        parts.append(group)

    df = pd.concat(
        parts,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Evento futuro
    # --------------------------------------------------------

    df[
        "irrigation_event_next_24h"
    ] = np.where(
        df["irrigation_depth_next_24h"].notna(),
        (
            df["irrigation_depth_next_24h"]
            > 0
        ).astype(int),
        np.nan
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GERAÇÃO DO DATASET COM PREVISÃO METEOROLÓGICA")
    print("=" * 70)

    # --------------------------------------------------------
    # Verificação dos arquivos
    # --------------------------------------------------------

    check_file(DATASET_3_PATH)
    check_file(NASA_PATH)
    check_file(FORECAST_PATH)

    # --------------------------------------------------------
    # Dataset 3
    # --------------------------------------------------------

    print("\nCarregando Dataset 3...")

    df = pd.read_csv(
        DATASET_3_PATH
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    print(
        f"Registros Dataset 3: {len(df):,}"
    )

    # --------------------------------------------------------
    # Restrição temporal
    # --------------------------------------------------------

    df = df[
        (df["timestamp"] >= "2024-01-01")
        &
        (df["timestamp"] <= "2025-12-31 23:00:00")
    ].copy()

    print(
        f"Registros após filtro temporal: {len(df):,}"
    )

    # --------------------------------------------------------
    # Features históricas
    # --------------------------------------------------------

    print("\nCriando features históricas...")

    df = add_historical_features(
        df
    )

    # --------------------------------------------------------
    # NASA POWER
    # --------------------------------------------------------

    print("\nCarregando dados NASA POWER...")

    nasa = pd.read_csv(
        NASA_PATH
    )

    nasa["timestamp"] = pd.to_datetime(
        nasa["timestamp"]
    )

    # --------------------------------------------------------
    # Seleção das variáveis meteorológicas
    # --------------------------------------------------------

    nasa_columns = [
        "timestamp",
        "temperature_c",
        "air_humidity_percent",
        "wind_speed_m_s",
        "solar_radiation_kwh_m2",
        "precipitation_mm",
    ]

    nasa_columns = [
        c for c in nasa_columns
        if c in nasa.columns
    ]

    nasa = nasa[nasa_columns].copy()

    # --------------------------------------------------------
    # Remove possíveis duplicatas
    # --------------------------------------------------------

    nasa = nasa.drop_duplicates(
        "timestamp"
    )

    # --------------------------------------------------------
    # Merge NASA
    # --------------------------------------------------------

    df = df.merge(
        nasa,
        on="timestamp",
        how="left",
        suffixes=("", "_nasa")
    )

    # --------------------------------------------------------
    # Carregar previsão
    # --------------------------------------------------------

    forecast = load_forecast()

    print(
        f"Registros de previsão: {len(forecast):,}"
    )

    # --------------------------------------------------------
    # Alinhar horizontes
    # --------------------------------------------------------

    forecast = align_forecasts(
        forecast
    )

    print(
        f"Registros após alinhamento: "
        f"{len(forecast):,}"
    )

    # --------------------------------------------------------
    # Merge das previsões
    # --------------------------------------------------------

    df = df.merge(
        forecast,
        on="timestamp",
        how="left"
    )

    # --------------------------------------------------------
    # Radiação negativa -> NaN
    # --------------------------------------------------------

    radiation_forecast_columns = [
        "forecast_radiation_24h_w_m2",
        "forecast_radiation_48h_w_m2",
        "forecast_radiation_72h_w_m2",
    ]

    for column in radiation_forecast_columns:

        if column in df.columns:

            df.loc[
                df[column] < 0,
                column
            ] = np.nan

    # --------------------------------------------------------
    # Features agregadas da previsão
    # --------------------------------------------------------

    print(
        "\nCriando features agregadas "
        "das previsões..."
    )

    df = add_forecast_features(
        df
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    df = create_targets(
        df
    )

    # --------------------------------------------------------
    # Remover linhas sem forecast completo
    # --------------------------------------------------------

    forecast_columns = [
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
    ]

    forecast_columns = [
        c for c in forecast_columns
        if c in df.columns
    ]

    before_forecast_filter = len(df)

    df = df.dropna(
        subset=forecast_columns
    ).copy()

    print(
        f"\nRemovidas por falta de previsão: "
        f"{before_forecast_filter - len(df):,}"
    )

    # --------------------------------------------------------
    # Remover linhas sem target válido
    # --------------------------------------------------------

    before_target_filter = len(df)

    df = df[
        df["irrigation_depth_next_24h"]
        .notna()
    ].copy()

    print(
        f"Removidas sem janela futura "
        f"de 24h válida: "
        f"{before_target_filter - len(df):,}"
    )

    # --------------------------------------------------------
    # Recalcular evento depois do filtro
    # --------------------------------------------------------

    df[
        "irrigation_event_next_24h"
    ] = (
        df["irrigation_depth_next_24h"]
        > 0
    ).astype(int)

    # --------------------------------------------------------
    # Ordenação final
    # --------------------------------------------------------

    df = df.sort_values(
        ["scenario_id", "timestamp"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Salvar
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # RESUMO
    # ========================================================

    print("\n" + "=" * 70)
    print("DATASET GERADO COM SUCESSO")
    print("=" * 70)

    print(
        f"Arquivo: {OUTPUT_PATH}"
    )

    print(
        f"Registros: {len(df):,}"
    )

    print(
        f"Colunas: {len(df.columns)}"
    )

    print(
        f"Primeiro timestamp: "
        f"{df['timestamp'].min()}"
    )

    print(
        f"Último timestamp: "
        f"{df['timestamp'].max()}"
    )

    print(
        f"Eventos futuros: "
        f"{df['irrigation_event_next_24h'].sum():,.0f}"
    )

    print(
        f"Taxa de evento: "
        f"{df['irrigation_event_next_24h'].mean() * 100:.2f}%"
    )

    print(
        f"Lâmina média futura: "
        f"{df['irrigation_depth_next_24h'].mean():.4f} mm"
    )

    print(
        f"Lâmina mediana futura: "
        f"{df['irrigation_depth_next_24h'].median():.4f} mm"
    )

    print(
        f"Lâmina máxima futura: "
        f"{df['irrigation_depth_next_24h'].max():.4f} mm"
    )

    # --------------------------------------------------------
    # Verificação de NaN nas previsões
    # --------------------------------------------------------

    print("\nCobertura das previsões:")

    for column in forecast_columns:

        missing = df[column].isna().sum()

        print(
            f"  {column}: "
            f"{len(df) - missing:,}/"
            f"{len(df):,}"
        )

    print("\nConcluído.")


if __name__ == "__main__":
    main()