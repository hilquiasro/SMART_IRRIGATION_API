import requests
import pandas as pd

# ============================================================
# CONFIGURAÇÃO
# ============================================================

# Santo Amaro - Bahia
latitude = -12.5467
longitude = -38.7119

start_date = "20200101"
end_date = "20251231"

# Parâmetros horários da NASA POWER
parameters = [
    "T2M",
    "RH2M",
    "WS10M",
    "ALLSKY_SFC_SW_DWN",
    "PRECTOTCORR",
]

# ============================================================
# URL DA NASA POWER
# ============================================================

url = (
    "https://power.larc.nasa.gov/api/temporal/hourly/point"
    f"?parameters={','.join(parameters)}"
    "&community=AG"
    f"&longitude={longitude}"
    f"&latitude={latitude}"
    f"&start={start_date}"
    f"&end={end_date}"
    "&format=JSON"
)

# ============================================================
# DOWNLOAD
# ============================================================

print("Baixando dados horários da NASA POWER...")
print(f"Localização: {latitude}, {longitude}")
print(f"Período: {start_date} até {end_date}")
print()

response = requests.get(
    url,
    timeout=120
)

if response.status_code != 200:
    print("Erro ao consultar a NASA POWER.")
    print("Status code:", response.status_code)
    print()
    print("Resposta:")
    print(response.text)

    raise SystemExit(1)

# ============================================================
# PROCESSAMENTO
# ============================================================

data = response.json()

parameters_data = data["properties"]["parameter"]

df = pd.DataFrame(parameters_data)

# O índice vem no formato YYYYMMDDHH
df.index = pd.to_datetime(
    df.index,
    format="%Y%m%d%H"
)

df.index.name = "timestamp"

df.reset_index(
    inplace=True
)

# ============================================================
# RENOMEAR COLUNAS
# ============================================================

df.rename(
    columns={
        "T2M": "temperature_c",
        "RH2M": "air_humidity_percent",
        "WS10M": "wind_speed_m_s",
        "ALLSKY_SFC_SW_DWN": "solar_radiation_kwh_m2",
        "PRECTOTCORR": "precipitation_mm",
    },
    inplace=True
)

# ============================================================
# ORGANIZAÇÃO
# ============================================================

df.sort_values(
    "timestamp",
    inplace=True
)

df.reset_index(
    drop=True,
    inplace=True
)

# ============================================================
# VERIFICAÇÃO
# ============================================================

print("Verificando os dados...")
print()

print("Valores ausentes:")
print(df.isnull().sum())

print()

print("Quantidade de registros:")
print(f"{len(df):,}")

print()

print("Período:")
print("Início:", df["timestamp"].min())
print("Fim:", df["timestamp"].max())

print()

print("Colunas:")
print(df.columns.tolist())

print()

print("Primeiras linhas:")
print(df.head())

print()

print("Estatísticas:")
print(df.describe())

# ============================================================
# SALVAR
# ============================================================

output = "ml/data/weather_santo_amaro_2020_2025_hourly.csv"

df.to_csv(
    output,
    index=False
)

# ============================================================
# RESULTADO
# ============================================================

print()
print("=" * 60)
print("DOWNLOAD CONCLUÍDO!")
print("=" * 60)
print()
print("Arquivo salvo em:")
print(output)
print()
print(f"Registros: {len(df):,}")
print(f"Colunas: {len(df.columns)}")
print()
print("Arquivo pronto para a próxima etapa.")