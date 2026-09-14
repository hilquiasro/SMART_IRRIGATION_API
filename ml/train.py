from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


# ============================================================
# CONFIGURAÇÃO
# ============================================================

RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "irrigation_dataset.csv"

RESULTS_DIR = BASE_DIR / "results_2"
MODELS_DIR = BASE_DIR / "models_2"

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


# ============================================================
# FUNÇÕES
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Calcula métricas gerais e específicas dos eventos.
    """

    y_pred = np.clip(y_pred, 0, None)

    # --------------------------------------------------------
    # Métricas gerais
    # --------------------------------------------------------

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    # --------------------------------------------------------
    # Eventos reais
    # --------------------------------------------------------

    event_mask = y_true > 0

    y_true_events = y_true[event_mask]
    y_pred_events = y_pred[event_mask]

    if len(y_true_events) > 0:

        event_mae = mean_absolute_error(
            y_true_events,
            y_pred_events
        )

        event_rmse = np.sqrt(
            mean_squared_error(
                y_true_events,
                y_pred_events
            )
        )

        event_r2 = r2_score(
            y_true_events,
            y_pred_events
        )

    else:

        event_mae = np.nan
        event_rmse = np.nan
        event_r2 = np.nan

    # --------------------------------------------------------
    # Detecção de eventos
    # --------------------------------------------------------

    actual_event = (y_true > 0).astype(int)

    predicted_event = (y_pred > 0.1).astype(int)

    precision = precision_score(
        actual_event,
        predicted_event,
        zero_division=0
    )

    recall = recall_score(
        actual_event,
        predicted_event,
        zero_division=0
    )

    f1 = f1_score(
        actual_event,
        predicted_event,
        zero_division=0
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "event_mae": event_mae,
        "event_rmse": event_rmse,
        "event_r2": event_r2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "real_events": int(actual_event.sum()),
        "predicted_events": int(predicted_event.sum()),
    }


# ============================================================
# CARREGAMENTO
# ============================================================

print("=" * 70)
print("EXPERIMENTO 2 - AJUSTE CONTROLADO DOS MODELOS")
print("=" * 70)

print()
print(f"Dataset: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print(f"Registros carregados: {len(df):,}")


# ============================================================
# VALIDAÇÃO
# ============================================================

required_columns = FEATURES + [TARGET, "timestamp"]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Colunas ausentes: {missing_columns}"
    )


df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

if df["timestamp"].isna().any():
    raise ValueError(
        "Existem timestamps inválidos."
    )


if df[required_columns].isna().any().any():
    raise ValueError(
        "Existem valores ausentes."
    )


df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# DIVISÃO TEMPORAL
# ============================================================

train_df = df[
    df["timestamp"] < "2025-01-01"
].copy()

test_df = df[
    df["timestamp"] >= "2025-01-01"
].copy()


print()
print("Divisão temporal:")
print(f"Treino: {len(train_df):,} registros")
print(f"Teste : {len(test_df):,} registros")


X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ============================================================
# PREPROCESSAMENTO
# ============================================================

numeric_features = [
    "soil_moisture_percent",
    "temperature_c",
    "air_humidity_percent",
    "solar_radiation_kwh_m2",
    "hour_sin",
    "hour_cos",
]

categorical_features = [
    "crop_stage"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
    ]
)


# ============================================================
# MODELOS
# ============================================================

models = {}


# ------------------------------------------------------------
# Modelos sem ajuste
# ------------------------------------------------------------

models["linear_regression"] = LinearRegression()

models["gradient_boosting"] = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=RANDOM_SEED,
    loss="squared_error"
)


# ------------------------------------------------------------
# Decision Tree
# ------------------------------------------------------------

for leaf in [1, 2, 5, 10, 20]:

    models[
        f"decision_tree_leaf_{leaf}"
    ] = DecisionTreeRegressor(
        random_state=RANDOM_SEED,
        max_depth=12,
        min_samples_leaf=leaf
    )


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

for leaf in [1, 2, 5, 10, 20]:

    models[
        f"random_forest_leaf_{leaf}"
    ] = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        min_samples_leaf=leaf
    )


# ------------------------------------------------------------
# Extra Trees
# ------------------------------------------------------------

for leaf in [1, 2, 5, 10, 20]:

    models[
        f"extra_trees_leaf_{leaf}"
    ] = ExtraTreesRegressor(
        n_estimators=300,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        min_samples_leaf=leaf
    )


# ============================================================
# BASELINE
# ============================================================

print()
print("-" * 70)
print("BASELINE")
print("-" * 70)

baseline_prediction = np.zeros(len(y_test))

baseline_metrics = calculate_metrics(
    y_test.values,
    baseline_prediction
)

print(
    f"MAE geral       : "
    f"{baseline_metrics['mae']:.6f}"
)

print(
    f"RMSE geral      : "
    f"{baseline_metrics['rmse']:.6f}"
)

print(
    f"R² geral        : "
    f"{baseline_metrics['r2']:.6f}"
)

print(
    f"MAE eventos     : "
    f"{baseline_metrics['event_mae']:.6f}"
)

print(
    f"RMSE eventos    : "
    f"{baseline_metrics['event_rmse']:.6f}"
)


results = [
    {
        "model": "baseline_zero",
        **baseline_metrics
    }
]


# ============================================================
# TREINAMENTO
# ============================================================

predictions = {}

for name, model in models.items():

    print()
    print("-" * 70)
    print(f"Treinando: {name}")
    print("-" * 70)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(X_test)

    y_pred = np.clip(
        y_pred,
        0,
        None
    )

    metrics = calculate_metrics(
        y_test.values,
        y_pred
    )

    results.append(
        {
            "model": name,
            **metrics
        }
    )

    predictions[name] = y_pred

    print(
        f"MAE geral       : "
        f"{metrics['mae']:.6f}"
    )

    print(
        f"RMSE geral      : "
        f"{metrics['rmse']:.6f}"
    )

    print(
        f"R² geral        : "
        f"{metrics['r2']:.6f}"
    )

    print(
        f"MAE eventos     : "
        f"{metrics['event_mae']:.6f}"
    )

    print(
        f"RMSE eventos    : "
        f"{metrics['event_rmse']:.6f}"
    )

    print(
        f"Precisão eventos: "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall eventos  : "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F1 eventos      : "
        f"{metrics['f1']:.6f}"
    )

    print(
        f"Eventos reais   : "
        f"{metrics['real_events']}"
    )

    print(
        f"Eventos previstos: "
        f"{metrics['predicted_events']}"
    )


# ============================================================
# RESULTADOS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "mae"
).reset_index(drop=True)


print()
print("=" * 70)
print("COMPARAÇÃO DOS MODELOS")
print("=" * 70)

print(
    results_df[
        [
            "model",
            "mae",
            "rmse",
            "r2",
            "event_mae",
            "event_rmse",
            "precision",
            "recall",
            "f1",
        ]
    ].to_string(index=False)
)


# ============================================================
# SALVAR COMPARAÇÃO
# ============================================================

comparison_file = (
    RESULTS_DIR /
    "model_comparison.csv"
)

results_df.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# ESCOLHA DO MELHOR MODELO
# ============================================================

# Para esta rodada, mantemos o critério objetivo
# de menor MAE geral para permitir comparação
# direta com o Experimento 1.

non_baseline = results_df[
    results_df["model"] != "baseline_zero"
]

best_row = non_baseline.iloc[
    non_baseline["mae"].argmin()
]

best_model_name = best_row["model"]

best_prediction = predictions[
    best_model_name
]


print()
print("=" * 70)
print("MELHOR MODELO PELO MAE GERAL")
print("=" * 70)

print(
    f"Modelo: {best_model_name}"
)

print(
    f"MAE: {best_row['mae']:.6f}"
)

print(
    f"RMSE: {best_row['rmse']:.6f}"
)

print(
    f"R²: {best_row['r2']:.6f}"
)

print(
    f"MAE eventos: "
    f"{best_row['event_mae']:.6f}"
)

print(
    f"F1 eventos: "
    f"{best_row['f1']:.6f}"
)


# ============================================================
# TREINAR NOVAMENTE O MELHOR MODELO
# ============================================================

best_base_model = models[
    best_model_name
]

best_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            best_base_model
        ),
    ]
)

best_pipeline.fit(
    X_train,
    y_train
)


model_file = (
    MODELS_DIR /
    "irrigation_model_experiment_2.joblib"
)

joblib.dump(
    best_pipeline,
    model_file
)


# ============================================================
# PREVISÕES DO MELHOR MODELO
# ============================================================

prediction_df = test_df[
    [
        "timestamp",
        TARGET
    ]
].copy()

prediction_df[
    "predicted_irrigation_depth_mm"
] = best_prediction

prediction_file = (
    RESULTS_DIR /
    "best_model_predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# GRÁFICOS
# ============================================================

plot_df = results_df[
    results_df["model"] != "baseline_zero"
].copy()


def save_bar_plot(
    column,
    title,
    filename,
    ylabel
):

    ordered = plot_df.sort_values(
        column
    )

    plt.figure(figsize=(12, 6))

    plt.bar(
        ordered["model"],
        ordered[column]
    )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / filename,
        dpi=300
    )

    plt.close()


save_bar_plot(
    "mae",
    "Comparação dos modelos - MAE geral",
    "mae_comparison.png",
    "MAE (mm)"
)


save_bar_plot(
    "rmse",
    "Comparação dos modelos - RMSE geral",
    "rmse_comparison.png",
    "RMSE (mm)"
)


save_bar_plot(
    "r2",
    "Comparação dos modelos - R² geral",
    "r2_comparison.png",
    "R²"
)


save_bar_plot(
    "event_mae",
    "Comparação dos modelos - MAE nos eventos",
    "event_mae_comparison.png",
    "MAE dos eventos (mm)"
)


save_bar_plot(
    "f1",
    "Comparação dos modelos - F1 dos eventos",
    "f1_comparison.png",
    "F1"
)


# ============================================================
# REAL VS PREDITO
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    best_prediction,
    alpha=0.25,
    s=10
)

max_value = max(
    y_test.max(),
    best_prediction.max()
)

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--"
)

plt.xlabel(
    "Irrigação real (mm)"
)

plt.ylabel(
    "Irrigação prevista (mm)"
)

plt.title(
    f"Valores reais vs previstos - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ============================================================
# RESÍDUOS
# ============================================================

residuals = (
    y_test.values -
    best_prediction
)

plt.figure(figsize=(10, 6))

plt.scatter(
    best_prediction,
    residuals,
    alpha=0.25,
    s=10
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Irrigação prevista (mm)"
)

plt.ylabel(
    "Resíduo (mm)"
)

plt.title(
    f"Resíduos - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "residuals.png",
    dpi=300
)

plt.close()


# ============================================================
# PREVISÕES AO LONGO DO TEMPO
# ============================================================

first_30_days = (
    test_df["timestamp"]
    <= test_df["timestamp"].min()
    + pd.Timedelta(days=30)
)

plt.figure(figsize=(14, 6))

plt.plot(
    test_df.loc[
        first_30_days,
        "timestamp"
    ],
    y_test.loc[
        first_30_days
    ],
    label="Real"
)

plt.plot(
    test_df.loc[
        first_30_days,
        "timestamp"
    ],
    best_prediction[
        first_30_days.to_numpy()
    ],
    label="Previsto"
)

plt.xlabel("Data")
plt.ylabel("Irrigação (mm)")

plt.title(
    f"Irrigação real vs prevista - "
    f"primeiros 30 dias de 2025"
)

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "predictions_over_time.png",
    dpi=300
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("EXPERIMENTO 2 CONCLUÍDO")
print("=" * 70)

print()
print("Resultados:")
print(comparison_file)

print()
print("Modelo:")
print(model_file)

print()
print("Previsões:")
print(prediction_file)

print()
print("Gráficos salvos em:")
print(RESULTS_DIR)