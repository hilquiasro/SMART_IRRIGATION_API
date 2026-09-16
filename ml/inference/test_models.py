from pathlib import Path
import json

import joblib
import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results_forecast_current"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CAMINHOS DOS DATASETS
# ============================================================

DATASET_V8 = DATA_DIR / "irrigation_dataset.csv"

DATASET_FORECAST = (
    DATA_DIR / "irrigation_dataset_forecast.csv"
)

DATASET_FORECAST_CURRENT = (
    DATA_DIR / "irrigation_dataset_forecast_current.csv"
)


# ============================================================
# CAMINHOS DOS MODELOS
# ============================================================

# V8
V8_CLASSIFIER = (
    MODELS_DIR
    / "model_v8"
    / "classifier_mlp.joblib"
)

V8_REGRESSOR = (
    MODELS_DIR
    / "model_v8"
    / "regressor_extra_trees.joblib"
)


# Forecast-24h
FORECAST_CLASSIFIER = (
    MODELS_DIR
    / "model_forecast"
    / "classifier_random_forest.joblib"
)

FORECAST_REGRESSOR = (
    MODELS_DIR
    / "model_forecast"
    / "regressor_extra_trees.joblib"
)


# Forecast-Current
FORECAST_CURRENT_CLASSIFIER = (
    MODELS_DIR
    / "model_forecast_current"
    / "classifier_HistGradientBoosting.joblib"
)

FORECAST_CURRENT_REGRESSOR = (
    MODELS_DIR
    / "model_forecast_current"
    / "regressor_LinearRegression.joblib"
)


# ============================================================
# THRESHOLDS
# ============================================================

THRESHOLD_V8 = 0.43

THRESHOLD_FORECAST = 0.41

THRESHOLD_FORECAST_CURRENT = 0.16


# ============================================================
# FEATURES DO V8
# ============================================================

# ============================================================
# FEATURES DO V8
# ============================================================

FEATURES_V8 = [
    # Variáveis atuais
    "soil_moisture_percent",
    "temperature_c",
    "air_humidity_percent",
    "wind_speed_m_s",
    "solar_radiation_kwh_m2",
    "precipitation_mm",
    "crop_stage",

    # Memória / histórico
    "soil_moisture_lag_1h",
    "soil_moisture_lag_3h",
    "soil_moisture_lag_6h",

    # Demanda hídrica
    "kc",
    "et0",
    "etc",
    "etc_6h",
    "etc_24h",

    # Chuva recente
    "rain_6h",
    "rain_24h",

    # Condições recentes
    "temperature_6h_mean",
    "radiation_6h_sum",

    # Ciclicidade temporal
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

# ============================================================
# FEATURES DOS MODELOS COM PREVISÃO
# ============================================================

FEATURES_FORECAST = [
    # Histórico / condições atuais
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

    # Previsão +24h
    "forecast_temperature_24h_c",
    "forecast_humidity_24h_percent",
    "forecast_precipitation_24h_mm",
    "forecast_wind_24h_m_s",
    "forecast_radiation_24h_w_m2",
    "forecast_et0_24h_mm",

    # Previsão +48h
    "forecast_temperature_48h_c",
    "forecast_humidity_48h_percent",
    "forecast_precipitation_48h_mm",
    "forecast_wind_48h_m_s",
    "forecast_radiation_48h_w_m2",
    "forecast_et0_48h_mm",

    # Previsão +72h
    "forecast_temperature_72h_c",
    "forecast_humidity_72h_percent",
    "forecast_precipitation_72h_mm",
    "forecast_wind_72h_m_s",
    "forecast_radiation_72h_w_m2",
    "forecast_et0_72h_mm",

    # Agregados das previsões
    "forecast_rain_total_72h",
    "forecast_temperature_mean_72h",
    "forecast_humidity_mean_72h",
    "forecast_wind_mean_72h",
    "forecast_radiation_mean_72h",
    "forecast_et0_mean_72h",
]


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def check_file(path: Path):
    """
    Verifica se um arquivo existe.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"\nArquivo não encontrado:\n{path}"
        )

    print(f" ✓ {path}")


def load_model(path: Path):
    """
    Carrega um modelo joblib.
    """
    check_file(path)

    model = joblib.load(path)

    print(f" ✓ Modelo carregado: {path.name}")

    return model


def check_features(df: pd.DataFrame, features: list, dataset_name: str):
    """
    Verifica se todas as features necessárias existem.
    """
    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:
        raise KeyError(
            f"\nFeatures ausentes no dataset {dataset_name}:\n"
            + "\n".join(f"  - {feature}" for feature in missing)
        )


def make_prediction(
    classifier,
    regressor,
    row: pd.Series,
    features: list,
    threshold: float,
):
    """
    Faz a inferência em uma única amostra.

    Primeiro:
        classificador -> decide se deve irrigar.

    Depois:
        regressor -> estima a lâmina.

    Se não houver irrigação:
        lâmina prevista = 0.
    """

    x = row[features].to_frame().T

    probability = float(
        classifier.predict_proba(x)[0, 1]
    )

    irrigation = probability >= threshold

    if irrigation:
        predicted_depth = float(
            regressor.predict(x)[0]
        )

        # Evita lâmina negativa por segurança.
        predicted_depth = max(
            0.0,
            predicted_depth
        )
    else:
        predicted_depth = 0.0

    return {
        "probability": probability,
        "threshold": threshold,
        "irrigation_decision": bool(irrigation),
        "predicted_depth_mm": predicted_depth,
    }


# ============================================================
# INÍCIO
# ============================================================

print()
print("=" * 70)
print("TESTE DE INFERÊNCIA DOS MODELOS")
print("=" * 70)


# ============================================================
# 1. CARREGAR MODELOS
# ============================================================

print()
print("=" * 70)
print("CARREGANDO MODELOS")
print("=" * 70)

print()
print("V8:")

v8_classifier = load_model(V8_CLASSIFIER)
v8_regressor = load_model(V8_REGRESSOR)


print()
print("Forecast-24h:")

forecast_classifier = load_model(
    FORECAST_CLASSIFIER
)

forecast_regressor = load_model(
    FORECAST_REGRESSOR
)


print()
print("Forecast-Current:")

forecast_current_classifier = load_model(
    FORECAST_CURRENT_CLASSIFIER
)

forecast_current_regressor = load_model(
    FORECAST_CURRENT_REGRESSOR
)


# ============================================================
# 2. CARREGAR DATASETS
# ============================================================

print()
print("=" * 70)
print("CARREGANDO DATASETS")
print("=" * 70)

check_file(DATASET_V8)
check_file(DATASET_FORECAST)
check_file(DATASET_FORECAST_CURRENT)


print()
print("Carregando V8...")

df_v8 = pd.read_csv(
    DATASET_V8,
    parse_dates=["timestamp"]
)

print(
    f"  V8: {len(df_v8):,} registros"
)


print()
print("Carregando Forecast-24h...")

df_forecast = pd.read_csv(
    DATASET_FORECAST,
    parse_dates=["timestamp"]
)

print(
    f"  Forecast-24h: {len(df_forecast):,} registros"
)


print()
print("Carregando Forecast-Current...")

df_forecast_current = pd.read_csv(
    DATASET_FORECAST_CURRENT,
    parse_dates=["timestamp"]
)

print(
    f"  Forecast-Current: "
    f"{len(df_forecast_current):,} registros"
)


# ============================================================
# 3. VERIFICAR FEATURES
# ============================================================

print()
print("=" * 70)
print("VERIFICANDO FEATURES")
print("=" * 70)

check_features(
    df_v8,
    FEATURES_V8,
    "V8"
)

check_features(
    df_forecast,
    FEATURES_FORECAST,
    "Forecast-24h"
)

check_features(
    df_forecast_current,
    FEATURES_FORECAST,
    "Forecast-Current"
)

print()
print(
    f" ✓ V8: {len(FEATURES_V8)} features"
)

print(
    f" ✓ Forecast-24h: {len(FEATURES_FORECAST)} features"
)

print(
    f" ✓ Forecast-Current: {len(FEATURES_FORECAST)} features"
)


# ============================================================
# 4. PADRONIZAR CHAVES
# ============================================================

print()
print("=" * 70)
print("PREPARANDO CHAVES DE COMPARAÇÃO")
print("=" * 70)

# O dataset possui vários cenários para o mesmo timestamp.
# Portanto, NÃO podemos usar timestamp sozinho.

for df in [
    df_v8,
    df_forecast,
    df_forecast_current,
]:

    df["scenario_id"] = df["scenario_id"].astype(int)


# ============================================================
# 5. ENCONTRAR AMOSTRA COMUM
# ============================================================

print()
print("=" * 70)
print("PROCURANDO AMOSTRA COMUM")
print("=" * 70)

common_keys = (
    df_v8[
        ["timestamp", "scenario_id"]
    ]
    .merge(
        df_forecast[
            ["timestamp", "scenario_id"]
        ],
        on=[
            "timestamp",
            "scenario_id",
        ],
        how="inner",
    )
    .merge(
        df_forecast_current[
            ["timestamp", "scenario_id"]
        ],
        on=[
            "timestamp",
            "scenario_id",
        ],
        how="inner",
    )
)


if common_keys.empty:

    raise RuntimeError(
        "\nNenhuma combinação comum de "
        "timestamp + scenario_id foi encontrada."
    )


# Preferir uma amostra de 2025.
common_2025 = common_keys[
    common_keys["timestamp"].dt.year == 2025
]


if not common_2025.empty:

    common_keys = common_2025


# Ordenar para obter uma amostra determinística.

common_keys = common_keys.sort_values(
    [
        "timestamp",
        "scenario_id",
    ]
).reset_index(drop=True)


sample = common_keys.iloc[0]

timestamp = sample["timestamp"]

scenario_id = int(
    sample["scenario_id"]
)


print()
print(
    f" ✓ {len(common_keys):,} combinações comuns encontradas."
)

print()
print(
    "Amostra escolhida:"
)

print(
    f"  Timestamp: {timestamp}"
)

print(
    f"  Cenário: {scenario_id}"
)


# ============================================================
# 6. LOCALIZAR AS LINHAS
# ============================================================

row_v8_df = df_v8[
    (df_v8["timestamp"] == timestamp)
    & (
        df_v8["scenario_id"]
        == scenario_id
    )
]

row_forecast_df = df_forecast[
    (df_forecast["timestamp"] == timestamp)
    & (
        df_forecast["scenario_id"]
        == scenario_id
    )
]

row_forecast_current_df = (
    df_forecast_current[
        (
            df_forecast_current["timestamp"]
            == timestamp
        )
        & (
            df_forecast_current["scenario_id"]
            == scenario_id
        )
    ]
)


if len(row_v8_df) != 1:
    raise RuntimeError(
        f"V8 retornou {len(row_v8_df)} "
        "linhas para a chave selecionada."
    )


if len(row_forecast_df) != 1:
    raise RuntimeError(
        f"Forecast-24h retornou "
        f"{len(row_forecast_df)} linhas "
        "para a chave selecionada."
    )


if len(row_forecast_current_df) != 1:
    raise RuntimeError(
        f"Forecast-Current retornou "
        f"{len(row_forecast_current_df)} linhas "
        "para a chave selecionada."
    )


row_v8 = row_v8_df.iloc[0]

row_forecast = row_forecast_df.iloc[0]

row_forecast_current = (
    row_forecast_current_df.iloc[0]
)


# ============================================================
# 7. INFERÊNCIA V8
# ============================================================

print()
print("=" * 70)
print("V8")
print("=" * 70)

v8_result = make_prediction(
    classifier=v8_classifier,
    regressor=v8_regressor,
    row=row_v8,
    features=FEATURES_V8,
    threshold=THRESHOLD_V8,
)

v8_real_depth = float(
    row_v8["irrigation_depth_mm"]
)

v8_real_event = (
    v8_real_depth > 0
)

print()
print(
    f"Probabilidade de irrigação: "
    f"{v8_result['probability']:.6f}"
)

print(
    f"Threshold: "
    f"{v8_result['threshold']:.2f}"
)

print(
    f"Decisão: "
    f"{'IRRIGAR' if v8_result['irrigation_decision'] else 'NÃO IRRIGAR'}"
)

print(
    f"Lâmina prevista: "
    f"{v8_result['predicted_depth_mm']:.6f} mm"
)

print(
    f"Lâmina real: "
    f"{v8_real_depth:.6f} mm"
)

print(
    f"Evento real: "
    f"{'SIM' if v8_real_event else 'NÃO'}"
)


# ============================================================
# 8. INFERÊNCIA FORECAST-24H
# ============================================================

print()
print("=" * 70)
print("FORECAST-24H")
print("=" * 70)

forecast_result = make_prediction(
    classifier=forecast_classifier,
    regressor=forecast_regressor,
    row=row_forecast,
    features=FEATURES_FORECAST,
    threshold=THRESHOLD_FORECAST,
)

forecast_real_depth = float(
    row_forecast["irrigation_depth_next_24h"]
)

forecast_real_event = (
    forecast_real_depth > 0
)

print()
print(
    f"Probabilidade de irrigação: "
    f"{forecast_result['probability']:.6f}"
)

print(
    f"Threshold: "
    f"{forecast_result['threshold']:.2f}"
)

print(
    f"Decisão: "
    f"{'IRRIGAR' if forecast_result['irrigation_decision'] else 'NÃO IRRIGAR'}"
)

print(
    f"Lâmina prevista: "
    f"{forecast_result['predicted_depth_mm']:.6f} mm"
)

print(
    f"Lâmina real nas próximas 24h: "
    f"{forecast_real_depth:.6f} mm"
)

print(
    f"Evento real nas próximas 24h: "
    f"{'SIM' if forecast_real_event else 'NÃO'}"
)


# ============================================================
# 9. INFERÊNCIA FORECAST-CURRENT
# ============================================================

print()
print("=" * 70)
print("FORECAST-CURRENT")
print("=" * 70)

forecast_current_result = make_prediction(
    classifier=forecast_current_classifier,
    regressor=forecast_current_regressor,
    row=row_forecast_current,
    features=FEATURES_FORECAST,
    threshold=THRESHOLD_FORECAST_CURRENT,
)

forecast_current_real_depth = float(
    row_forecast_current[
        "irrigation_depth_current"
    ]
)

forecast_current_real_event = (
    forecast_current_real_depth > 0
)

print()
print(
    f"Probabilidade de irrigação: "
    f"{forecast_current_result['probability']:.6f}"
)

print(
    f"Threshold: "
    f"{forecast_current_result['threshold']:.2f}"
)

print(
    f"Decisão: "
    f"{'IRRIGAR' if forecast_current_result['irrigation_decision'] else 'NÃO IRRIGAR'}"
)

print(
    f"Lâmina prevista: "
    f"{forecast_current_result['predicted_depth_mm']:.6f} mm"
)

print(
    f"Lâmina real: "
    f"{forecast_current_real_depth:.6f} mm"
)

print(
    f"Evento real: "
    f"{'SIM' if forecast_current_real_event else 'NÃO'}"
)


# ============================================================
# 10. COMPARAÇÃO
# ============================================================

print()
print("=" * 70)
print("COMPARAÇÃO")
print("=" * 70)

print()

print(
    f"{'Modelo':<22}"
    f"{'Prob.':>12}"
    f"{'Decisão':>15}"
    f"{'Previsto (mm)':>18}"
    f"{'Real (mm)':>15}"
)

print("-" * 82)


print(
    f"{'V8':<22}"
    f"{v8_result['probability']:>12.4f}"
    f"{'IRRIGAR' if v8_result['irrigation_decision'] else 'NÃO IRRIGAR':>15}"
    f"{v8_result['predicted_depth_mm']:>18.4f}"
    f"{v8_real_depth:>15.4f}"
)


print(
    f"{'Forecast-24h':<22}"
    f"{forecast_result['probability']:>12.4f}"
    f"{'IRRIGAR' if forecast_result['irrigation_decision'] else 'NÃO IRRIGAR':>15}"
    f"{forecast_result['predicted_depth_mm']:>18.4f}"
    f"{forecast_real_depth:>15.4f}"
)


print(
    f"{'Forecast-Current':<22}"
    f"{forecast_current_result['probability']:>12.4f}"
    f"{'IRRIGAR' if forecast_current_result['irrigation_decision'] else 'NÃO IRRIGAR':>15}"
    f"{forecast_current_result['predicted_depth_mm']:>18.4f}"
    f"{forecast_current_real_depth:>15.4f}"
)


# ============================================================
# 11. SALVAR RESULTADO
# ============================================================

output = {
    "timestamp": timestamp.isoformat(),
    "scenario_id": scenario_id,

    "v8": {
        "features_count": len(FEATURES_V8),
        "threshold": THRESHOLD_V8,
        "probability": v8_result["probability"],
        "decision": v8_result[
            "irrigation_decision"
        ],
        "predicted_depth_mm": (
            v8_result[
                "predicted_depth_mm"
            ]
        ),
        "real_depth_mm": v8_real_depth,
        "real_event": v8_real_event,
    },

    "forecast_24h": {
        "features_count": len(FEATURES_FORECAST),
        "threshold": THRESHOLD_FORECAST,
        "probability": (
            forecast_result[
                "probability"
            ]
        ),
        "decision": (
            forecast_result[
                "irrigation_decision"
            ]
        ),
        "predicted_depth_mm": (
            forecast_result[
                "predicted_depth_mm"
            ]
        ),
        "real_depth_next_24h_mm": (
            forecast_real_depth
        ),
        "real_event_next_24h": (
            forecast_real_event
        ),
    },

    "forecast_current": {
        "features_count": len(FEATURES_FORECAST),
        "threshold": (
            THRESHOLD_FORECAST_CURRENT
        ),
        "probability": (
            forecast_current_result[
                "probability"
            ]
        ),
        "decision": (
            forecast_current_result[
                "irrigation_decision"
            ]
        ),
        "predicted_depth_mm": (
            forecast_current_result[
                "predicted_depth_mm"
            ]
        ),
        "real_depth_mm": (
            forecast_current_real_depth
        ),
        "real_event": (
            forecast_current_real_event
        ),
    },
}


output_path = (
    RESULTS_DIR
    / "inference_test.json"
)


with open(
    output_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        output,
        file,
        ensure_ascii=False,
        indent=4,
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("TESTE CONCLUÍDO")
print("=" * 70)

print()
print(
    f"Resultado salvo em:\n"
    f"{output_path}"
)

print()