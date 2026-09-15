import requests
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURAÇÕES
# ============================================================

LATITUDE = -12.5467
LONGITUDE = -38.7119

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

TIMEZONE = "America/Bahia"

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

OUTPUT_FILE = Path(
    "ml/data/forecast_santo_amaro_2024_2025.csv"
)


# ============================================================
# VARIÁVEIS DA PREVISÃO
# ============================================================

VARIABLES = [
    "temperature_2m_previous_day1",
    "relative_humidity_2m_previous_day1",
    "precipitation_previous_day1",
    "wind_speed_10m_previous_day1",
    "shortwave_radiation_previous_day1",
    "et0_fao_evapotranspiration_previous_day1",

    "temperature_2m_previous_day2",
    "relative_humidity_2m_previous_day2",
    "precipitation_previous_day2",
    "wind_speed_10m_previous_day2",
    "shortwave_radiation_previous_day2",
    "et0_fao_evapotranspiration_previous_day2",

    "temperature_2m_previous_day3",
    "relative_humidity_2m_previous_day3",
    "precipitation_previous_day3",
    "wind_speed_10m_previous_day3",
    "shortwave_radiation_previous_day3",
    "et0_fao_evapotranspiration_previous_day3",
]


# ============================================================
# NOMES FINAIS
# ============================================================

RENAME = {
    "temperature_2m_previous_day1":
        "forecast_temperature_1d_c",
    "relative_humidity_2m_previous_day1":
        "forecast_humidity_1d_percent",
    "precipitation_previous_day1":
        "forecast_precipitation_1d_mm",
    "wind_speed_10m_previous_day1":
        "forecast_wind_1d_m_s",
    "shortwave_radiation_previous_day1":
        "forecast_radiation_1d_w_m2",
    "et0_fao_evapotranspiration_previous_day1":
        "forecast_et0_1d_mm",

    "temperature_2m_previous_day2":
        "forecast_temperature_2d_c",
    "relative_humidity_2m_previous_day2":
        "forecast_humidity_2d_percent",
    "precipitation_previous_day2":
        "forecast_precipitation_2d_mm",
    "wind_speed_10m_previous_day2":
        "forecast_wind_2d_m_s",
    "shortwave_radiation_previous_day2":
        "forecast_radiation_2d_w_m2",
    "et0_fao_evapotranspiration_previous_day2":
        "forecast_et0_2d_mm",

    "temperature_2m_previous_day3":
        "forecast_temperature_3d_c",
    "relative_humidity_2m_previous_day3":
        "forecast_humidity_3d_percent",
    "precipitation_previous_day3":
        "forecast_precipitation_3d_mm",
    "wind_speed_10m_previous_day3":
        "forecast_wind_3d_m_s",
    "shortwave_radiation_previous_day3":
        "forecast_radiation_3d_w_m2",
    "et0_fao_evapotranspiration_previous_day3":
        "forecast_et0_3d_mm",
}


# ============================================================
# DOWNLOAD
# ============================================================

def download_forecast():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(VARIABLES),
        "timezone": TIMEZONE,
    }

    print("=" * 60)
    print("DOWNLOAD DE PREVISÕES METEOROLÓGICAS")
    print("=" * 60)

    print(f"Período: {START_DATE} → {END_DATE}")
    print(f"Local: {LATITUDE}, {LONGITUDE}")
    print()

    response = requests.get(
        URL,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise RuntimeError(
            "A API não retornou dados horários."
        )

    df = pd.DataFrame(data["hourly"])

    print(
        f"Registros recebidos: {len(df):,}"
    )

    # ========================================================
    # TRATAMENTO
    # ========================================================

    df = df.rename(columns=RENAME)

    df["timestamp"] = pd.to_datetime(
        df["time"]
    )

    df = df.drop(columns=["time"])

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    forecast_columns = list(RENAME.values())

    # Converte todas as previsões para número
    for column in forecast_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # ========================================================
    # REMOÇÃO DE REGISTROS SEM OS 3 HORIZONTES
    # ========================================================

    before = len(df)

    df = df.dropna(
        subset=forecast_columns
    ).copy()

    removed = before - len(df)

    # ========================================================
    # VALIDAÇÕES
    # ========================================================

    duplicate_timestamps = (
        df["timestamp"].duplicated().sum()
    )

    if duplicate_timestamps > 0:
        raise RuntimeError(
            "Foram encontradas duplicatas de timestamp."
        )

    missing = df[forecast_columns].isna().sum().sum()

    if missing > 0:
        raise RuntimeError(
            "Ainda existem valores ausentes."
        )

    # ========================================================
    # SALVA
    # ========================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # RESUMO
    # ========================================================

    print()
    print("=" * 60)
    print("TRATAMENTO CONCLUÍDO")
    print("=" * 60)

    print(
        f"Registros antes:     {before:,}"
    )

    print(
        f"Registros removidos: {removed:,}"
    )

    print(
        f"Registros finais:    {len(df):,}"
    )

    print()
    print(
        f"Período final: "
        f"{df['timestamp'].min()} → "
        f"{df['timestamp'].max()}"
    )

    print()
    print(
        "Valores ausentes: 0"
    )

    print(
        f"Arquivo salvo em:\n{OUTPUT_FILE}"
    )

    print()
    print("Primeiras linhas:")
    print(df.head())

    print()
    print("Últimas linhas:")
    print(df.tail())


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    download_forecast()