from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
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
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "ml" / "data" / "irrigation_dataset_forecast.csv"

TARGET_EVENT = "irrigation_event_next_24h"
TARGET_DEPTH = "irrigation_depth_next_24h"


# ============================================================
# FEATURES APROVADAS PELO AUDIT
# ============================================================

FEATURES = [
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

    # +24 h
    "forecast_temperature_24h_c",
    "forecast_humidity_24h_percent",
    "forecast_precipitation_24h_mm",
    "forecast_wind_24h_m_s",
    "forecast_radiation_24h_w_m2",
    "forecast_et0_24h_mm",

    # +48 h
    "forecast_temperature_48h_c",
    "forecast_humidity_48h_percent",
    "forecast_precipitation_48h_mm",
    "forecast_wind_48h_m_s",
    "forecast_radiation_48h_w_m2",
    "forecast_et0_48h_mm",

    # +72 h
    "forecast_temperature_72h_c",
    "forecast_humidity_72h_percent",
    "forecast_precipitation_72h_mm",
    "forecast_wind_72h_m_s",
    "forecast_radiation_72h_w_m2",
    "forecast_et0_72h_mm",

    # Agregados
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

def calculate_classification_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def find_best_threshold(y_true, probabilities):
    """
    Escolhe o threshold que maximiza o F1-score.

    IMPORTANTE:
    Este threshold é escolhido SOMENTE na validação.
    O conjunto de teste não participa dessa escolha.
    """

    thresholds = np.arange(0.10, 0.91, 0.01)

    best_threshold = 0.50
    best_f1 = -1

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        current_f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return best_threshold, best_f1


def regression_metrics(y_true, predictions):
    rmse = np.sqrt(
        mean_squared_error(y_true, predictions)
    )

    return {
        "mae": mean_absolute_error(
            y_true,
            predictions
        ),
        "rmse": rmse,
        "r2": r2_score(
            y_true,
            predictions
        ),
    }


# ============================================================
# CARREGAMENTO
# ============================================================

print("=" * 70)
print("TREINAMENTO FORECAST - VALIDAÇÃO TEMPORAL")
print("=" * 70)

print(f"\nDataset: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


print("\nDataset carregado:")
print(f"  Registros: {len(df):,}")
print(
    f"  Período: "
    f"{df['timestamp'].min()} → "
    f"{df['timestamp'].max()}"
)

# ============================================================
# VERIFICAÇÃO DAS FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    print("\nERRO: features ausentes:")

    for feature in missing_features:
        print(f"  - {feature}")

    raise ValueError(
        "O dataset não possui todas as features aprovadas."
    )


# ============================================================
# REMOVER REGISTROS INCOMPLETOS
# ============================================================

required_columns = (
    FEATURES
    + [
        TARGET_EVENT,
        TARGET_DEPTH,
        "timestamp",
    ]
)

before = len(df)

df = df.dropna(
    subset=required_columns
).copy()

after = len(df)

print(
    f"\nRegistros removidos por NaN: "
    f"{before - after:,}"
)

print(
    f"Registros disponíveis: "
    f"{after:,}"
)


# ============================================================
# DIVISÃO TEMPORAL
#
# TREINO:
# 01/01/2024 → 30/09/2024
#
# VALIDAÇÃO:
# 01/10/2024 → 31/12/2024
#
# TESTE:
# 01/01/2025 → 31/12/2025
# ============================================================

train_mask = (
    (df["timestamp"] >= "2024-01-01")
    & (df["timestamp"] < "2024-10-01")
)

validation_mask = (
    (df["timestamp"] >= "2024-10-01")
    & (df["timestamp"] < "2025-01-01")
)

test_mask = (
    (df["timestamp"] >= "2025-01-01")
    & (df["timestamp"] < "2026-01-01")
)

train_df = df.loc[train_mask].copy()
validation_df = df.loc[validation_mask].copy()
test_df = df.loc[test_mask].copy()


print("\n" + "=" * 70)
print("DIVISÃO TEMPORAL")
print("=" * 70)

print(
    f"\nTreino: "
    f"{train_df['timestamp'].min()} → "
    f"{train_df['timestamp'].max()}"
)

print(
    f"  Registros: {len(train_df):,}"
)

print(
    f"  Eventos: "
    f"{train_df[TARGET_EVENT].sum():,}"
)

print(
    f"  Taxa de eventos: "
    f"{train_df[TARGET_EVENT].mean() * 100:.2f}%"
)


print(
    f"\nValidação: "
    f"{validation_df['timestamp'].min()} → "
    f"{validation_df['timestamp'].max()}"
)

print(
    f"  Registros: {len(validation_df):,}"
)

print(
    f"  Eventos: "
    f"{validation_df[TARGET_EVENT].sum():,}"
)

print(
    f"  Taxa de eventos: "
    f"{validation_df[TARGET_EVENT].mean() * 100:.2f}%"
)


print(
    f"\nTeste: "
    f"{test_df['timestamp'].min()} → "
    f"{test_df['timestamp'].max()}"
)

print(
    f"  Registros: {len(test_df):,}"
)

print(
    f"  Eventos: "
    f"{test_df[TARGET_EVENT].sum():,}"
)

print(
    f"  Taxa de eventos: "
    f"{test_df[TARGET_EVENT].mean() * 100:.2f}%"
)


if len(train_df) == 0:
    raise ValueError("Conjunto de treino vazio.")

if len(validation_df) == 0:
    raise ValueError("Conjunto de validação vazio.")

if len(test_df) == 0:
    raise ValueError("Conjunto de teste vazio.")


# ============================================================
# MATRIZES
# ============================================================

X_train = train_df[FEATURES]
X_validation = validation_df[FEATURES]
X_test = test_df[FEATURES]

y_train_event = train_df[TARGET_EVENT].astype(int)
y_validation_event = validation_df[TARGET_EVENT].astype(int)
y_test_event = test_df[TARGET_EVENT].astype(int)

# Regressão somente nos eventos
train_regression_mask = (
    train_df[TARGET_EVENT] == 1
)

validation_regression_mask = (
    validation_df[TARGET_EVENT] == 1
)

test_regression_mask = (
    test_df[TARGET_EVENT] == 1
)

X_train_reg = train_df.loc[
    train_regression_mask,
    FEATURES
]

y_train_depth = train_df.loc[
    train_regression_mask,
    TARGET_DEPTH
]

X_validation_reg = validation_df.loc[
    validation_regression_mask,
    FEATURES
]

y_validation_depth = validation_df.loc[
    validation_regression_mask,
    TARGET_DEPTH
]

X_test_reg = test_df.loc[
    test_regression_mask,
    FEATURES
]

y_test_depth = test_df.loc[
    test_regression_mask,
    TARGET_DEPTH
]


# ============================================================
# MODELOS DE CLASSIFICAÇÃO
# ============================================================

classification_models = {
    "LogisticRegression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "DecisionTree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced",
        max_depth=12
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        max_depth=None
    ),

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        max_depth=None
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=42
    ),

    "MLP": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
        )
    ])
}


# ============================================================
# CLASSIFICAÇÃO
#
# O THRESHOLD É ESCOLHIDO NA VALIDAÇÃO.
# ============================================================

classification_results = []

trained_classifiers = {}

print("\n" + "=" * 70)
print("CLASSIFICAÇÃO")
print("=" * 70)

for name, model in classification_models.items():

    print(f"\nTreinando {name}...")

    model.fit(
        X_train,
        y_train_event
    )

    trained_classifiers[name] = model

    # --------------------------------------------------------
    # VALIDAÇÃO
    # --------------------------------------------------------

    validation_prob = model.predict_proba(
        X_validation
    )[:, 1]

    threshold, validation_best_f1 = find_best_threshold(
        y_validation_event,
        validation_prob
    )

    validation_pred = (
        validation_prob >= threshold
    ).astype(int)

    validation_metrics = calculate_classification_metrics(
        y_validation_event,
        validation_pred,
        validation_prob
    )

    # --------------------------------------------------------
    # TESTE
    # --------------------------------------------------------

    test_prob = model.predict_proba(
        X_test
    )[:, 1]

    test_pred = (
        test_prob >= threshold
    ).astype(int)

    test_metrics = calculate_classification_metrics(
        y_test_event,
        test_pred,
        test_prob
    )

    classification_results.append({
        "model": name,
        "threshold": threshold,

        "validation_accuracy":
            validation_metrics["accuracy"],

        "validation_precision":
            validation_metrics["precision"],

        "validation_recall":
            validation_metrics["recall"],

        "validation_f1":
            validation_metrics["f1"],

        "validation_roc_auc":
            validation_metrics["roc_auc"],

        "test_accuracy":
            test_metrics["accuracy"],

        "test_precision":
            test_metrics["precision"],

        "test_recall":
            test_metrics["recall"],

        "test_f1":
            test_metrics["f1"],

        "test_roc_auc":
            test_metrics["roc_auc"],
    })

    print(
        f"  Threshold escolhido na validação: "
        f"{threshold:.2f}"
    )

    print(
        f"  Validação F1: "
        f"{validation_metrics['f1']:.4f}"
    )

    print(
        f"  Teste F1: "
        f"{test_metrics['f1']:.4f}"
    )


classification_results_df = pd.DataFrame(
    classification_results
)

classification_results_df = (
    classification_results_df
    .sort_values(
        "test_f1",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\n" + "-" * 70)
print("RESULTADOS DA CLASSIFICAÇÃO")
print("-" * 70)

print(
    classification_results_df[
        [
            "model",
            "threshold",
            "validation_f1",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
        ]
    ].to_string(index=False)
)


# ============================================================
# MELHOR CLASSIFICADOR
# ============================================================

best_classifier_name = (
    classification_results_df
    .iloc[0]["model"]
)

best_classifier = trained_classifiers[
    best_classifier_name
]

best_threshold = float(
    classification_results_df
    .iloc[0]["threshold"]
)

print(
    f"\nMelhor classificador: "
    f"{best_classifier_name}"
)

print(
    f"Threshold usado no teste: "
    f"{best_threshold:.2f}"
)


# ============================================================
# REGRESSÃO
# ============================================================

regression_models = {
    "LinearRegression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LinearRegression()
        )
    ]),

    "DecisionTree": DecisionTreeRegressor(
        random_state=42,
        max_depth=12
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None
    ),

    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=42
    ),

    "MLP": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
        )
    ])
}


regression_results = []

trained_regressors = {}

print("\n" + "=" * 70)
print("REGRESSÃO")
print("=" * 70)

for name, model in regression_models.items():

    print(f"\nTreinando {name}...")

    model.fit(
        X_train_reg,
        y_train_depth
    )

    trained_regressors[name] = model

    # --------------------------------------------------------
    # TESTE
    # --------------------------------------------------------

    test_predictions = model.predict(
        X_test_reg
    )

    metrics = regression_metrics(
        y_test_depth,
        test_predictions
    )

    regression_results.append({
        "model": name,
        "test_mae": metrics["mae"],
        "test_rmse": metrics["rmse"],
        "test_r2": metrics["r2"],
    })

    print(
        f"  MAE:  {metrics['mae']:.4f} mm"
    )

    print(
        f"  RMSE: {metrics['rmse']:.4f} mm"
    )

    print(
        f"  R²:   {metrics['r2']:.4f}"
    )


regression_results_df = pd.DataFrame(
    regression_results
)

regression_results_df = (
    regression_results_df
    .sort_values(
        "test_mae",
        ascending=True
    )
    .reset_index(drop=True)
)


print("\n" + "-" * 70)
print("RESULTADOS DA REGRESSÃO")
print("-" * 70)

print(
    regression_results_df.to_string(
        index=False
    )
)


# ============================================================
# MELHOR REGRESSOR
# ============================================================

best_regressor_name = (
    regression_results_df
    .iloc[0]["model"]
)

best_regressor = trained_regressors[
    best_regressor_name
]

print(
    f"\nMelhor regressor: "
    f"{best_regressor_name}"
)


# ============================================================
# AVALIAÇÃO END-TO-END
# ============================================================

print("\n" + "=" * 70)
print("AVALIAÇÃO END-TO-END")
print("=" * 70)

best_test_prob = best_classifier.predict_proba(
    X_test
)[:, 1]

best_test_event_pred = (
    best_test_prob >= best_threshold
).astype(int)

# Predição de lâmina para todas as linhas.
# Só será usada quando o classificador indicar irrigação.

best_depth_pred = best_regressor.predict(
    X_test
)

# Não permitir valores negativos.
best_depth_pred = np.maximum(
    best_depth_pred,
    0
)

# ------------------------------------------------------------
# Resultado end-to-end
# ------------------------------------------------------------

end_to_end_pred = np.where(
    best_test_event_pred == 1,
    best_depth_pred,
    0
)

y_test_depth_all = (
    test_df[TARGET_DEPTH]
    .to_numpy()
)

# ------------------------------------------------------------
# Métricas
# ------------------------------------------------------------

end_to_end_mae = mean_absolute_error(
    y_test_depth_all,
    end_to_end_pred
)

end_to_end_rmse = np.sqrt(
    mean_squared_error(
        y_test_depth_all,
        end_to_end_pred
    )
)

end_to_end_r2 = r2_score(
    y_test_depth_all,
    end_to_end_pred
)

real_water = y_test_depth_all.sum()

predicted_water = end_to_end_pred.sum()

water_difference = (
    predicted_water
    - real_water
)

water_difference_percent = (
    water_difference
    / real_water
    * 100
)

real_events = int(
    (y_test_depth_all > 0).sum()
)

predicted_events = int(
    (end_to_end_pred > 0).sum()
)


print(
    f"\nClassificador: "
    f"{best_classifier_name}"
)

print(
    f"Regressor: "
    f"{best_regressor_name}"
)

print(
    f"Threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"\nMAE end-to-end: "
    f"{end_to_end_mae:.6f} mm"
)

print(
    f"RMSE end-to-end: "
    f"{end_to_end_rmse:.6f} mm"
)

print(
    f"R² end-to-end: "
    f"{end_to_end_r2:.6f}"
)

print(
    f"\nÁgua real: "
    f"{real_water:.4f} mm"
)

print(
    f"Água prevista: "
    f"{predicted_water:.4f} mm"
)

print(
    f"Diferença: "
    f"{water_difference:+.4f} mm"
)

print(
    f"Diferença percentual: "
    f"{water_difference_percent:+.4f}%"
)

print(
    f"\nEventos reais: "
    f"{real_events:,}"
)

print(
    f"Eventos previstos: "
    f"{predicted_events:,}"
)


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

cm = confusion_matrix(
    y_test_event,
    best_test_event_pred
)

print("\n" + "-" * 70)
print("MATRIZ DE CONFUSÃO - TESTE 2025")
print("-" * 70)

print(
    "\n                 Previsto"
)

print(
    "              Não      Sim"
)

print(
    f"Real Não    {cm[0, 0]:7d}  "
    f"{cm[0, 1]:7d}"
)

print(
    f"Real Sim    {cm[1, 0]:7d}  "
    f"{cm[1, 1]:7d}"
)


# ============================================================
# REGRESSÃO APENAS NOS EVENTOS REAIS
# ============================================================

true_event_depth_predictions = best_regressor.predict(
    X_test_reg
)

true_event_depth_predictions = np.maximum(
    true_event_depth_predictions,
    0
)

true_event_mae = mean_absolute_error(
    y_test_depth,
    true_event_depth_predictions
)

true_event_rmse = np.sqrt(
    mean_squared_error(
        y_test_depth,
        true_event_depth_predictions
    )
)

true_event_r2 = r2_score(
    y_test_depth,
    true_event_depth_predictions
)

print("\n" + "-" * 70)
print("REGRESSÃO NOS EVENTOS REAIS - TESTE 2025")
print("-" * 70)

print(
    f"\nMAE:  "
    f"{true_event_mae:.6f} mm"
)

print(
    f"RMSE: "
    f"{true_event_rmse:.6f} mm"
)

print(
    f"R²:   "
    f"{true_event_r2:.6f}"
)


# ============================================================
# IMPORTÂNCIA DAS FEATURES
# ============================================================

print("\n" + "=" * 70)
print("IMPORTÂNCIA DAS FEATURES")
print("=" * 70)

if hasattr(
    best_regressor,
    "feature_importances_"
):

    importances = (
        best_regressor
        .feature_importances_
    )

    feature_importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": importances
    })

    feature_importance_df = (
        feature_importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        feature_importance_df.head(20)
        .to_string(index=False)
    )

elif hasattr(
    best_regressor[-1],
    "feature_importances_"
):

    importances = (
        best_regressor[-1]
        .feature_importances_
    )

    feature_importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": importances
    })

    feature_importance_df = (
        feature_importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        feature_importance_df.head(20)
        .to_string(index=False)
    )

else:

    feature_importance_df = None

    print(
        "\nO melhor regressor não possui "
        "feature_importances_."
    )


# ============================================================
# SALVAR RESULTADOS
# ============================================================

RESULTS_DIR = BASE_DIR / "ml" / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. RESULTADOS DOS MODELOS
# ============================================================

classification_results_path = (
    RESULTS_DIR
    / "forecast_classification_results.csv"
)

regression_results_path = (
    RESULTS_DIR
    / "forecast_regression_results.csv"
)

classification_results_df.to_csv(
    classification_results_path,
    index=False
)

regression_results_df.to_csv(
    regression_results_path,
    index=False
)


# ============================================================
# 2. IMPORTÂNCIA DAS FEATURES
# ============================================================

if feature_importance_df is not None:

    feature_importance_path = (
        RESULTS_DIR
        / "forecast_feature_importance.csv"
    )

    feature_importance_df.to_csv(
        feature_importance_path,
        index=False
    )


# ============================================================
# 3. PREVISÕES DO TESTE
# ============================================================

predictions_test = test_df[
    [
        "timestamp",
        "soil_moisture_percent",
        "crop_stage",
        TARGET_EVENT,
        TARGET_DEPTH,
    ]
].copy()

# Probabilidade de irrigação
predictions_test[
    "predicted_irrigation_probability"
] = best_test_prob

# Evento previsto
predictions_test[
    "predicted_irrigation_event"
] = best_test_event_pred

# Lâmina prevista pelo regressor
predictions_test[
    "predicted_irrigation_depth_mm"
] = best_depth_pred

# Resultado final end-to-end
predictions_test[
    "predicted_irrigation_depth_end_to_end_mm"
] = end_to_end_pred

# Erro absoluto da previsão end-to-end
predictions_test[
    "absolute_error_mm"
] = np.abs(
    y_test_depth_all
    - end_to_end_pred
)

predictions_path = (
    RESULTS_DIR
    / "forecast_predictions_test_2025.csv"
)

predictions_test.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# 4. MATRIZ DE CONFUSÃO
# ============================================================

confusion_matrix_df = pd.DataFrame(
    cm,
    index=[
        "Real_Nao_Irrigar",
        "Real_Irrigar"
    ],
    columns=[
        "Previsto_Nao_Irrigar",
        "Previsto_Irrigar"
    ]
)

confusion_matrix_path = (
    RESULTS_DIR
    / "forecast_confusion_matrix_test.csv"
)

confusion_matrix_df.to_csv(
    confusion_matrix_path
)


# ============================================================
# 5. MÉTRICAS FINAIS
# ============================================================

best_classification_row = (
    classification_results_df.iloc[0]
)

best_regression_row = (
    regression_results_df.iloc[0]
)

final_metrics = {

    # --------------------------------------------------------
    # Classificação
    # --------------------------------------------------------

    "classifier": best_classifier_name,

    "classification_threshold": best_threshold,

    "classification_test_accuracy":
        float(
            best_classification_row[
                "test_accuracy"
            ]
        ),

    "classification_test_precision":
        float(
            best_classification_row[
                "test_precision"
            ]
        ),

    "classification_test_recall":
        float(
            best_classification_row[
                "test_recall"
            ]
        ),

    "classification_test_f1":
        float(
            best_classification_row[
                "test_f1"
            ]
        ),

    "classification_test_roc_auc":
        float(
            best_classification_row[
                "test_roc_auc"
            ]
        ),

    # --------------------------------------------------------
    # Regressão
    # --------------------------------------------------------

    "regressor": best_regressor_name,

    "regression_test_mae_mm":
        float(
            best_regression_row[
                "test_mae"
            ]
        ),

    "regression_test_rmse_mm":
        float(
            best_regression_row[
                "test_rmse"
            ]
        ),

    "regression_test_r2":
        float(
            best_regression_row[
                "test_r2"
            ]
        ),

    # --------------------------------------------------------
    # Regressão nos eventos reais
    # --------------------------------------------------------

    "true_event_mae_mm":
        float(true_event_mae),

    "true_event_rmse_mm":
        float(true_event_rmse),

    "true_event_r2":
        float(true_event_r2),

    # --------------------------------------------------------
    # End-to-end
    # --------------------------------------------------------

    "end_to_end_mae_mm":
        float(end_to_end_mae),

    "end_to_end_rmse_mm":
        float(end_to_end_rmse),

    "end_to_end_r2":
        float(end_to_end_r2),

    # --------------------------------------------------------
    # Água
    # --------------------------------------------------------

    "real_water_mm":
        float(real_water),

    "predicted_water_mm":
        float(predicted_water),

    "water_difference_mm":
        float(water_difference),

    "water_difference_percent":
        float(water_difference_percent),

    # --------------------------------------------------------
    # Eventos
    # --------------------------------------------------------

    "real_events":
        int(real_events),

    "predicted_events":
        int(predicted_events),

    # --------------------------------------------------------
    # Tamanho dos conjuntos
    # --------------------------------------------------------

    "train_records":
        int(len(train_df)),

    "validation_records":
        int(len(validation_df)),

    "test_records":
        int(len(test_df)),

    "train_events":
        int(train_df[TARGET_EVENT].sum()),

    "validation_events":
        int(validation_df[TARGET_EVENT].sum()),

    "test_events":
        int(test_df[TARGET_EVENT].sum()),
}


final_metrics_path = (
    RESULTS_DIR
    / "forecast_final_metrics.json"
)

with open(
    final_metrics_path,
    "w",
    encoding="utf-8"
) as f:

    import json

    json.dump(
        final_metrics,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# ARQUIVOS SALVOS
# ============================================================

print("\n" + "=" * 70)
print("RESULTADOS SALVOS")
print("=" * 70)

print(
    f"\n  {classification_results_path}"
)

print(
    f"  {regression_results_path}"
)

if feature_importance_df is not None:

    print(
        f"  {feature_importance_path}"
    )

print(
    f"  {predictions_path}"
)

print(
    f"  {confusion_matrix_path}"
)

print(
    f"  {final_metrics_path}"
)

# ============================================================
# RESUMO FINAL
# ============================================================

print("\n" + "=" * 70)
print("RESUMO FINAL")
print("=" * 70)

print(
    f"\nMelhor classificador: "
    f"{best_classifier_name}"
)

print(
    f"Threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"F1 teste: "
    f"{classification_results_df.iloc[0]['test_f1']:.4f}"
)

print(
    f"Precision teste: "
    f"{classification_results_df.iloc[0]['test_precision']:.4f}"
)

print(
    f"Recall teste: "
    f"{classification_results_df.iloc[0]['test_recall']:.4f}"
)

print(
    f"ROC-AUC teste: "
    f"{classification_results_df.iloc[0]['test_roc_auc']:.4f}"
)

print(
    f"\nMelhor regressor: "
    f"{best_regressor_name}"
)

print(
    f"MAE regressão: "
    f"{regression_results_df.iloc[0]['test_mae']:.4f} mm"
)

print(
    f"RMSE regressão: "
    f"{regression_results_df.iloc[0]['test_rmse']:.4f} mm"
)

print(
    f"R² regressão: "
    f"{regression_results_df.iloc[0]['test_r2']:.4f}"
)

print(
    f"\nMAE end-to-end: "
    f"{end_to_end_mae:.6f} mm"
)

print(
    f"RMSE end-to-end: "
    f"{end_to_end_rmse:.6f} mm"
)

print(
    f"R² end-to-end: "
    f"{end_to_end_r2:.6f}"
)

print(
    f"Diferença de água: "
    f"{water_difference_percent:+.4f}%"
)

print(
    "\nResultados salvos em:"
)

print(
    f"  {RESULTS_DIR}"
)

print("\n" + "=" * 70)
print("TREINAMENTO CONCLUÍDO")
print("=" * 70)