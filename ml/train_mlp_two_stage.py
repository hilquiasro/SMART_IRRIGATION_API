from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "irrigation_dataset.csv"

RESULTS_DIR = BASE_DIR / "results_4"
MODELS_DIR = BASE_DIR / "models_4"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "soil_moisture_percent",
    "temperature_c",
    "air_humidity_percent",
    "solar_radiation_kwh_m2",
    "crop_stage",
    "hour_sin",
    "hour_cos",
]

TARGET = "irrigation_depth_mm"

NUMERIC_FEATURES = [
    "soil_moisture_percent",
    "temperature_c",
    "air_humidity_percent",
    "solar_radiation_kwh_m2",
    "hour_sin",
    "hour_cos",
]

CATEGORICAL_FEATURES = [
    "crop_stage",
]


# ============================================================
# FUNÇÕES
# ============================================================

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def calculate_regression_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


# ============================================================
# CARREGAMENTO
# ============================================================

print("\nCarregando dataset...")

df = pd.read_csv(INPUT_FILE)

required_columns = FEATURES + [TARGET, "timestamp"]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Colunas ausentes no dataset: {missing_columns}"
    )

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

if df[required_columns].isnull().any().any():
    raise ValueError("Existem valores nulos no dataset.")

df["irrigation_event"] = (
    df[TARGET] > 0
).astype(int)


# ============================================================
# DIVISÃO TEMPORAL
# ============================================================

train_df = df[
    df["timestamp"] < "2025-01-01"
].copy()

test_df = df[
    df["timestamp"] >= "2025-01-01"
].copy()


X_train = train_df[FEATURES]
X_test = test_df[FEATURES]

y_train = train_df[TARGET]
y_test = test_df[TARGET]

event_train = train_df["irrigation_event"]
event_test = test_df["irrigation_event"]


print("\nDivisão temporal:")
print(f"Treinamento: {len(train_df)}")
print(f"Teste:       {len(test_df)}")

print(
    f"\nEventos no treinamento: "
    f"{event_train.sum()}"
)

print(
    f"Eventos no teste: "
    f"{event_test.sum()}"
)


# ============================================================
# MLP CLASSIFICADOR
# ============================================================

print("\nTreinando MLP Classificador...")

classifier_preprocessor = build_preprocessor()

mlp_classifier = Pipeline(
    steps=[
        (
            "preprocessor",
            classifier_preprocessor,
        ),
        (
            "model",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=RANDOM_SEED,
            ),
        ),
    ]
)

mlp_classifier.fit(
    X_train,
    event_train,
)


# ============================================================
# CLASSIFICAÇÃO
# ============================================================

event_probability = mlp_classifier.predict_proba(
    X_test
)[:, 1]

predicted_event = (
    event_probability >= 0.5
).astype(int)


classification_metrics = {
    "accuracy": accuracy_score(
        event_test,
        predicted_event,
    ),
    "precision": precision_score(
        event_test,
        predicted_event,
        zero_division=0,
    ),
    "recall": recall_score(
        event_test,
        predicted_event,
        zero_division=0,
    ),
    "f1": f1_score(
        event_test,
        predicted_event,
        zero_division=0,
    ),
    "roc_auc": roc_auc_score(
        event_test,
        event_probability,
    ),
}


print("\n" + "=" * 60)
print("MLP CLASSIFICADOR")
print("=" * 60)

for metric, value in classification_metrics.items():
    print(f"{metric:12s}: {value:.6f}")

print(
    f"\nEventos reais:     {event_test.sum()}"
)

print(
    f"Eventos previstos: {predicted_event.sum()}"
)


# ============================================================
# MLP REGRESSOR
# ============================================================

print("\nTreinando MLP Regressor...")

train_events = train_df[
    train_df[TARGET] > 0
].copy()

test_events = test_df[
    test_df[TARGET] > 0
].copy()


X_train_events = train_events[FEATURES]
y_train_events = train_events[TARGET]

X_test_events = test_events[FEATURES]
y_test_events = test_events[TARGET]


mlp_regressor_preprocessor = build_preprocessor()

mlp_regressor = Pipeline(
    steps=[
        (
            "preprocessor",
            mlp_regressor_preprocessor,
        ),
        (
            "model",
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=RANDOM_SEED,
            ),
        ),
    ]
)

mlp_regressor.fit(
    X_train_events,
    y_train_events,
)


# ============================================================
# REGRESSÃO NOS EVENTOS REAIS
# ============================================================

event_depth_prediction = mlp_regressor.predict(
    X_test_events
)

event_depth_prediction = np.clip(
    event_depth_prediction,
    0,
    None,
)


regression_metrics = calculate_regression_metrics(
    y_test_events,
    event_depth_prediction,
)


print("\n" + "=" * 60)
print("MLP REGRESSOR")
print("=" * 60)

for metric, value in regression_metrics.items():
    print(f"{metric:12s}: {value:.6f}")


# ============================================================
# MODELO END-TO-END
# ============================================================

print("\nCalculando desempenho end-to-end...")

predicted_depth = np.zeros(
    len(test_df),
    dtype=float,
)

predicted_indices = np.where(
    predicted_event == 1
)[0]

if len(predicted_indices) > 0:

    X_predicted_events = test_df.iloc[
        predicted_indices
    ][FEATURES]

    predicted_depth_values = (
        mlp_regressor.predict(
            X_predicted_events
        )
    )

    predicted_depth_values = np.clip(
        predicted_depth_values,
        0,
        None,
    )

    predicted_depth[
        predicted_indices
    ] = predicted_depth_values


# ============================================================
# MÉTRICAS END-TO-END
# ============================================================

overall_metrics = calculate_regression_metrics(
    y_test,
    predicted_depth,
)

event_metrics = calculate_regression_metrics(
    y_test[event_test == 1],
    predicted_depth[event_test == 1],
)


end_to_end_metrics = {
    "overall_mae": overall_metrics["mae"],
    "overall_rmse": overall_metrics["rmse"],
    "overall_r2": overall_metrics["r2"],
    "event_mae": event_metrics["mae"],
    "event_rmse": event_metrics["rmse"],
    "event_r2": event_metrics["r2"],
    "precision": classification_metrics["precision"],
    "recall": classification_metrics["recall"],
    "f1": classification_metrics["f1"],
    "roc_auc": classification_metrics["roc_auc"],
}


print("\n" + "=" * 60)
print("MLP END-TO-END")
print("=" * 60)

for metric, value in end_to_end_metrics.items():
    print(f"{metric:15s}: {value:.6f}")


# ============================================================
# MÉTRICAS DE ÁGUA
# ============================================================

actual_total_water = y_test.sum()

predicted_total_water = predicted_depth.sum()

water_bias = (
    predicted_total_water
    - actual_total_water
)

false_positives = (
    (predicted_event == 1)
    & (event_test == 0)
).sum()

false_negatives = (
    (predicted_event == 0)
    & (event_test == 1)
).sum()


print("\n" + "=" * 60)
print("ANÁLISE DE IRRIGAÇÃO")
print("=" * 60)

print(
    f"Água real:             "
    f"{actual_total_water:.3f} mm"
)

print(
    f"Água prevista:         "
    f"{predicted_total_water:.3f} mm"
)

print(
    f"Diferença:             "
    f"{water_bias:.3f} mm"
)

print(
    f"Falsos positivos:      "
    f"{false_positives}"
)

print(
    f"Falsos negativos:      "
    f"{false_negatives}"
)


# ============================================================
# SALVAR PREDIÇÕES
# ============================================================

predictions = pd.DataFrame({
    "timestamp": test_df["timestamp"].values,
    "actual_irrigation_depth_mm": y_test.values,
    "actual_event": event_test.values,
    "event_probability": event_probability,
    "predicted_event": predicted_event,
    "predicted_irrigation_depth_mm": predicted_depth,
})

predictions.to_csv(
    RESULTS_DIR / "mlp_predictions.csv",
    index=False,
)


# ============================================================
# SALVAR MÉTRICAS
# ============================================================

metrics_rows = []

for metric, value in classification_metrics.items():
    metrics_rows.append({
        "stage": "classifier",
        "metric": metric,
        "value": value,
    })

for metric, value in regression_metrics.items():
    metrics_rows.append({
        "stage": "event_regressor",
        "metric": metric,
        "value": value,
    })

for metric, value in end_to_end_metrics.items():
    metrics_rows.append({
        "stage": "end_to_end",
        "metric": metric,
        "value": value,
    })

metrics_rows.extend([
    {
        "stage": "water",
        "metric": "actual_total_water_mm",
        "value": actual_total_water,
    },
    {
        "stage": "water",
        "metric": "predicted_total_water_mm",
        "value": predicted_total_water,
    },
    {
        "stage": "water",
        "metric": "water_bias_mm",
        "value": water_bias,
    },
    {
        "stage": "water",
        "metric": "false_positives",
        "value": false_positives,
    },
    {
        "stage": "water",
        "metric": "false_negatives",
        "value": false_negatives,
    },
])

pd.DataFrame(metrics_rows).to_csv(
    RESULTS_DIR / "mlp_metrics.csv",
    index=False,
)


# ============================================================
# SALVAR MODELOS
# ============================================================

joblib.dump(
    mlp_classifier,
    MODELS_DIR / "mlp_classifier.joblib",
)

joblib.dump(
    mlp_regressor,
    MODELS_DIR / "mlp_regressor.joblib",
)


print("\n" + "=" * 60)
print("EXPERIMENTO 4 CONCLUÍDO")
print("=" * 60)

print(
    f"Resultados: {RESULTS_DIR}"
)

print(
    f"Modelos:    {MODELS_DIR}"
)