from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "irrigation_dataset_forecast_current.csv"
)

RESULTS_DIR = (
    BASE_DIR
    / "results_forecast_current"
)

MODELS_DIR = (
    BASE_DIR
    / "models"
    / "model_forecast_current"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FEATURES
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


TARGET_CLASSIFICATION = "irrigation_event_current"
TARGET_REGRESSION = "irrigation_depth_current"


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )


def find_best_threshold(
    y_true,
    probabilities,
):
    """
    Escolhe o threshold exclusivamente
    utilizando o conjunto de validação.
    """

    thresholds = np.arange(
        0.10,
        0.91,
        0.01,
    )

    best_threshold = 0.50
    best_f1 = -1

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return float(best_threshold)


def classification_metrics(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "threshold": float(threshold),

        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
    }

    if len(
        np.unique(y_true)
    ) == 2:

        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

    else:
        metrics["roc_auc"] = None

    return metrics, predictions


def regression_metrics(
    y_true,
    y_pred,
):

    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),

        "rmse": float(
            rmse(
                y_true,
                y_pred,
            )
        ),

        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
    }


# ============================================================
# INÍCIO
# ============================================================

print("=" * 70)
print("TREINAMENTO FORECAST-CURRENT")
print("=" * 70)


# ============================================================
# CARREGAMENTO
# ============================================================

print("\nCarregando dataset:")
print(DATASET_PATH)

df = pd.read_csv(
    DATASET_PATH,
    parse_dates=["timestamp"],
)

df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)

print(
    f"\nRegistros: {len(df):,}"
)

print(
    f"Colunas: {len(df.columns)}"
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

    raise RuntimeError(
        "Features ausentes no dataset:\n"
        + "\n".join(
            f"  - {feature}"
            for feature in missing_features
        )
    )


for target in [
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
]:

    if target not in df.columns:

        raise RuntimeError(
            f"Target ausente: {target}"
        )


print(
    f"\nFeatures encontradas: "
    f"{len(FEATURES)}"
)

print(
    "Targets encontrados: 2"
)


# ============================================================
# REMOÇÃO DE NaN
# ============================================================

required_columns = (
    FEATURES
    + [
        TARGET_CLASSIFICATION,
        TARGET_REGRESSION,
    ]
)

before = len(df)

df = df.dropna(
    subset=required_columns
).copy()

removed = before - len(df)

print(
    f"\nRegistros removidos por NaN: "
    f"{removed:,}"
)

print(
    f"Registros utilizados: "
    f"{len(df):,}"
)


# ============================================================
# TIPOS DOS TARGETS
# ============================================================

df[
    TARGET_CLASSIFICATION
] = (
    df[
        TARGET_CLASSIFICATION
    ].astype(int)
)

df[
    TARGET_REGRESSION
] = (
    df[
        TARGET_REGRESSION
    ].astype(float)
)


# ============================================================
# DIVISÃO TEMPORAL
# ============================================================

TRAIN_START = pd.Timestamp(
    "2024-01-01"
)

VALIDATION_START = pd.Timestamp(
    "2024-10-01"
)

TEST_START = pd.Timestamp(
    "2025-01-01"
)

TEST_END = pd.Timestamp(
    "2026-01-01"
)


train_df = df[
    (df["timestamp"] >= TRAIN_START)
    & (
        df["timestamp"]
        < VALIDATION_START
    )
].copy()


validation_df = df[
    (df["timestamp"] >= VALIDATION_START)
    & (
        df["timestamp"]
        < TEST_START
    )
].copy()


test_df = df[
    (df["timestamp"] >= TEST_START)
    & (
        df["timestamp"]
        < TEST_END
    )
].copy()


# ============================================================
# RESUMO DA DIVISÃO
# ============================================================

print("\n" + "=" * 70)
print("DIVISÃO TEMPORAL")
print("=" * 70)


for name, subset in [
    ("TREINO", train_df),
    ("VALIDAÇÃO", validation_df),
    ("TESTE", test_df),
]:

    events = subset[
        TARGET_CLASSIFICATION
    ].sum()

    event_rate = (
        events / len(subset)
        if len(subset) > 0
        else 0
    )

    print(
        f"\n{name}:"
    )

    print(
        f"  início: "
        f"{subset['timestamp'].min()}"
    )

    print(
        f"  fim:    "
        f"{subset['timestamp'].max()}"
    )

    print(
        f"  registros: "
        f"{len(subset):,}"
    )

    print(
        f"  eventos:   "
        f"{events:,}"
    )

    print(
        f"  taxa:      "
        f"{event_rate:.4%}"
    )


# ============================================================
# MATRIZES DE CLASSIFICAÇÃO
# ============================================================

X_train = train_df[
    FEATURES
]

X_validation = validation_df[
    FEATURES
]

X_test = test_df[
    FEATURES
]


y_train_class = train_df[
    TARGET_CLASSIFICATION
]

y_validation_class = validation_df[
    TARGET_CLASSIFICATION
]

y_test_class = test_df[
    TARGET_CLASSIFICATION
]


# ============================================================
# MODELOS DE CLASSIFICAÇÃO
# ============================================================

classification_models = {

    "LogisticRegression": Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]),

    "DecisionTree":
        DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        ),

    "RandomForest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),

    "ExtraTrees":
        ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),

    "HistGradientBoosting":
        HistGradientBoostingClassifier(
            random_state=42,
        ),

    "MLP": Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            MLPClassifier(
                hidden_layer_sizes=(
                    128,
                    64,
                ),
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
            ),
        ),
    ]),
}


# ============================================================
# TREINAMENTO DOS CLASSIFICADORES
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICAÇÃO")
print("=" * 70)


classification_results = {}

trained_classifiers = {}


for name, model in (
    classification_models.items()
):

    print(
        f"\nTreinando {name}..."
    )

    model.fit(
        X_train,
        y_train_class,
    )

    validation_probabilities = (
        model
        .predict_proba(
            X_validation
        )[:, 1]
    )

    threshold = (
        find_best_threshold(
            y_validation_class,
            validation_probabilities,
        )
    )

    validation_metrics, _ = (
        classification_metrics(
            y_validation_class,
            validation_probabilities,
            threshold,
        )
    )

    classification_results[
        name
    ] = {
        "validation":
            validation_metrics,
    }

    trained_classifiers[
        name
    ] = model

    print(
        f"  threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"  validation F1: "
        f"{validation_metrics['f1']:.4f}"
    )

    print(
        f"  validation ROC-AUC: "
        f"{validation_metrics['roc_auc']:.4f}"
    )


# ============================================================
# MELHOR CLASSIFICADOR
# ============================================================

best_classifier_name = max(
    classification_results,
    key=lambda name:
        classification_results[
            name
        ]["validation"]["f1"],
)

best_classifier = (
    trained_classifiers[
        best_classifier_name
    ]
)

best_threshold = (
    classification_results[
        best_classifier_name
    ]["validation"]["threshold"]
)


print("\n" + "=" * 70)
print("MELHOR CLASSIFICADOR")
print("=" * 70)

print(
    f"\nModelo: "
    f"{best_classifier_name}"
)

print(
    f"Threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Validation F1: "
    f"{classification_results[best_classifier_name]['validation']['f1']:.4f}"
)


# ============================================================
# TESTE DO CLASSIFICADOR
# ============================================================

test_probabilities = (
    best_classifier
    .predict_proba(
        X_test
    )[:, 1]
)

test_class_metrics, test_predictions = (
    classification_metrics(
        y_test_class,
        test_probabilities,
        best_threshold,
    )
)


classification_results[
    best_classifier_name
]["test"] = test_class_metrics


print("\n" + "=" * 70)
print("TESTE — CLASSIFICAÇÃO")
print("=" * 70)


for metric, value in (
    test_class_metrics.items()
):

    if metric == "threshold":
        continue

    if value is None:

        print(
            f"{metric}: N/A"
        )

    else:

        print(
            f"{metric}: "
            f"{value:.4f}"
        )


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

cm = confusion_matrix(
    y_test_class,
    test_predictions,
)

tn, fp, fn, tp = cm.ravel()


print("\nMatriz de confusão:")

print(
    f"  TN: {tn:,}"
)

print(
    f"  FP: {fp:,}"
)

print(
    f"  FN: {fn:,}"
)

print(
    f"  TP: {tp:,}"
)


# ============================================================
# REGRESSÃO
# ============================================================

print("\n" + "=" * 70)
print("REGRESSÃO")
print("=" * 70)


train_regression_df = train_df[
    train_df[
        TARGET_CLASSIFICATION
    ] == 1
].copy()


validation_regression_df = (
    validation_df[
        validation_df[
            TARGET_CLASSIFICATION
        ] == 1
    ].copy()
)


test_regression_df = test_df[
    test_df[
        TARGET_CLASSIFICATION
    ] == 1
].copy()


X_train_reg = (
    train_regression_df[
        FEATURES
    ]
)

y_train_reg = (
    train_regression_df[
        TARGET_REGRESSION
    ]
)


X_validation_reg = (
    validation_regression_df[
        FEATURES
    ]
)

y_validation_reg = (
    validation_regression_df[
        TARGET_REGRESSION
    ]
)


X_test_reg = (
    test_regression_df[
        FEATURES
    ]
)

y_test_reg = (
    test_regression_df[
        TARGET_REGRESSION
    ]
)


print(
    f"\nEventos usados no treino: "
    f"{len(train_regression_df):,}"
)

print(
    f"Eventos usados na validação: "
    f"{len(validation_regression_df):,}"
)

print(
    f"Eventos usados no teste: "
    f"{len(test_regression_df):,}"
)


# ============================================================
# MODELOS DE REGRESSÃO
# ============================================================

regression_models = {

    "LinearRegression": Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            LinearRegression(),
        ),
    ]),

    "DecisionTree":
        DecisionTreeRegressor(
            random_state=42,
        ),

    "RandomForest":
        RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),

    "ExtraTrees":
        ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),

    "HistGradientBoosting":
        HistGradientBoostingRegressor(
            random_state=42,
        ),

    "MLP": Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            MLPRegressor(
                hidden_layer_sizes=(
                    128,
                    64,
                ),
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
            ),
        ),
    ]),
}


# ============================================================
# TREINAMENTO DOS REGRESSORES
# ============================================================

regression_results = {}

trained_regressors = {}


for name, model in (
    regression_models.items()
):

    print(
        f"\nTreinando {name}..."
    )

    model.fit(
        X_train_reg,
        y_train_reg,
    )

    validation_predictions = (
        model.predict(
            X_validation_reg
        )
    )

    validation_predictions = np.clip(
        validation_predictions,
        0,
        None,
    )

    validation_metrics = (
        regression_metrics(
            y_validation_reg,
            validation_predictions,
        )
    )

    regression_results[
        name
    ] = {
        "validation":
            validation_metrics,
    }

    trained_regressors[
        name
    ] = model

    print(
        f"  validation MAE: "
        f"{validation_metrics['mae']:.4f} mm"
    )

    print(
        f"  validation RMSE: "
        f"{validation_metrics['rmse']:.4f} mm"
    )

    print(
        f"  validation R²: "
        f"{validation_metrics['r2']:.4f}"
    )


# ============================================================
# MELHOR REGRESSOR
# ============================================================

best_regressor_name = min(
    regression_results,
    key=lambda name:
        regression_results[
            name
        ]["validation"]["mae"],
)

best_regressor = (
    trained_regressors[
        best_regressor_name
    ]
)


print("\n" + "=" * 70)
print("MELHOR REGRESSOR")
print("=" * 70)

print(
    f"\nModelo: "
    f"{best_regressor_name}"
)

print(
    f"Validation MAE: "
    f"{regression_results[best_regressor_name]['validation']['mae']:.4f} mm"
)


# ============================================================
# TESTE DA REGRESSÃO
# ============================================================

test_regression_predictions = (
    best_regressor.predict(
        X_test_reg
    )
)

test_regression_predictions = np.clip(
    test_regression_predictions,
    0,
    None,
)


test_regression_metrics = (
    regression_metrics(
        y_test_reg,
        test_regression_predictions,
    )
)


regression_results[
    best_regressor_name
]["test"] = test_regression_metrics


print("\n" + "=" * 70)
print("TESTE — REGRESSÃO")
print("=" * 70)

print(
    f"\nMAE: "
    f"{test_regression_metrics['mae']:.4f} mm"
)

print(
    f"RMSE: "
    f"{test_regression_metrics['rmse']:.4f} mm"
)

print(
    f"R²: "
    f"{test_regression_metrics['r2']:.4f}"
)


# ============================================================
# COEFICIENTES DA REGRESSÃO LINEAR
# ============================================================
#
# Se LinearRegression for o melhor regressor,
# salvamos os coeficientes para análise.
#
# Como existe StandardScaler no pipeline,
# os coeficientes correspondem às features
# padronizadas.
#
# ============================================================

if (
    best_regressor_name
    == "LinearRegression"
):

    linear_model = (
        best_regressor
        .named_steps["model"]
    )

    coefficients = (
        linear_model.coef_
    )

    intercept = (
        linear_model.intercept_
    )

    coefficients_df = pd.DataFrame({

        "feature":
            FEATURES,

        "coefficient":
            coefficients,

        "absolute_coefficient":
            np.abs(coefficients),
    })

    coefficients_df = (
        coefficients_df
        .sort_values(
            "absolute_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    coefficients_df.to_csv(
        RESULTS_DIR
        / "regression_coefficients.csv",
        index=False,
    )

    with open(
        RESULTS_DIR
        / "regression_intercept.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "intercept": float(
                    intercept
                )
            },
            file,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("COEFICIENTES DA REGRESSÃO LINEAR")
    print("=" * 70)

    print(
        "\nTop 20 por magnitude:"
    )

    print(
        coefficients_df
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# END-TO-END
# ============================================================

print("\n" + "=" * 70)
print("AVALIAÇÃO END-TO-END")
print("=" * 70)


end_to_end_predictions = np.zeros(
    len(test_df)
)


predicted_irrigation_mask = (
    test_predictions == 1
)


if predicted_irrigation_mask.any():

    X_pred_reg = X_test[
        predicted_irrigation_mask
    ]

    predicted_depths = (
        best_regressor.predict(
            X_pred_reg
        )
    )

    predicted_depths = np.clip(
        predicted_depths,
        0,
        None,
    )

    end_to_end_predictions[
        predicted_irrigation_mask
    ] = predicted_depths


y_test_depth = (
    test_df[
        TARGET_REGRESSION
    ].to_numpy()
)


end_to_end_mae = (
    mean_absolute_error(
        y_test_depth,
        end_to_end_predictions,
    )
)


end_to_end_rmse = rmse(
    y_test_depth,
    end_to_end_predictions,
)


end_to_end_r2 = (
    r2_score(
        y_test_depth,
        end_to_end_predictions,
    )
)


real_water = (
    y_test_depth.sum()
)

predicted_water = (
    end_to_end_predictions.sum()
)


water_difference = (
    predicted_water
    - real_water
)


water_difference_percent = (
    water_difference
    / real_water
    * 100
    if real_water != 0
    else 0
)


real_events = (
    y_test_depth > 0
).sum()


predicted_events = (
    end_to_end_predictions > 0
).sum()


print(
    f"\nMAE: "
    f"{end_to_end_mae:.6f} mm"
)

print(
    f"RMSE: "
    f"{end_to_end_rmse:.6f} mm"
)

print(
    f"R²: "
    f"{end_to_end_r2:.6f}"
)

print(
    f"\nLâmina real total: "
    f"{real_water:.4f} mm"
)

print(
    f"Lâmina prevista total: "
    f"{predicted_water:.4f} mm"
)

print(
    f"Diferença: "
    f"{water_difference:.4f} mm"
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
# IMPORTÂNCIA DAS FEATURES
# ============================================================

feature_importance = None


if hasattr(
    best_regressor,
    "feature_importances_",
):

    feature_importance = (
        best_regressor
        .feature_importances_
    )


elif hasattr(
    best_regressor,
    "named_steps",
):

    inner_model = (
        best_regressor
        .named_steps["model"]
    )

    if hasattr(
        inner_model,
        "feature_importances_",
    ):

        feature_importance = (
            inner_model
            .feature_importances_
        )


if feature_importance is not None:

    importance_df = pd.DataFrame({

        "feature":
            FEATURES,

        "importance":
            feature_importance,
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_df.to_csv(
        RESULTS_DIR
        / "feature_importance.csv",
        index=False,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP 20 FEATURES"
    )

    print(
        "=" * 70
    )

    print(
        importance_df
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# SALVAR PREVISÕES
# ============================================================

predictions_df = test_df[
    [
        "timestamp",
        "scenario_id",
        TARGET_CLASSIFICATION,
        TARGET_REGRESSION,
    ]
].copy()


predictions_df[
    "predicted_event"
] = test_predictions


predictions_df[
    "predicted_event_probability"
] = test_probabilities


predictions_df[
    "predicted_depth_mm"
] = end_to_end_predictions


predictions_df.to_csv(
    RESULTS_DIR
    / "predictions_test_2025.csv",
    index=False,
)


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

pd.DataFrame(
    cm,
    index=[
        "real_0",
        "real_1",
    ],
    columns=[
        "pred_0",
        "pred_1",
    ],
).to_csv(
    RESULTS_DIR
    / "confusion_matrix_test.csv"
)


# ============================================================
# RESULTADOS COMPLETOS
# ============================================================

final_metrics = {

    "experiment":
        "forecast_current",

    "dataset":
        str(DATASET_PATH),

    "n_features":
        len(FEATURES),

    "features":
        FEATURES,

    "targets": {
        "classification":
            TARGET_CLASSIFICATION,

        "regression":
            TARGET_REGRESSION,
    },

    "split": {

        "train_start":
            str(TRAIN_START),

        "validation_start":
            str(VALIDATION_START),

        "test_start":
            str(TEST_START),

        "test_end":
            str(TEST_END),
    },

    "classification": {

        "validation":
            classification_results[
                best_classifier_name
            ]["validation"],

        "test":
            classification_results[
                best_classifier_name
            ]["test"],

        "best_model":
            best_classifier_name,
    },

    "regression": {

        "validation":
            regression_results[
                best_regressor_name
            ]["validation"],

        "test":
            regression_results[
                best_regressor_name
            ]["test"],

        "best_model":
            best_regressor_name,
    },

    "end_to_end": {

        "mae_mm":
            float(
                end_to_end_mae
            ),

        "rmse_mm":
            float(
                end_to_end_rmse
            ),

        "r2":
            float(
                end_to_end_r2
            ),

        "real_water_mm":
            float(
                real_water
            ),

        "predicted_water_mm":
            float(
                predicted_water
            ),

        "difference_mm":
            float(
                water_difference
            ),

        "difference_percent":
            float(
                water_difference_percent
            ),

        "real_events":
            int(real_events),

        "predicted_events":
            int(predicted_events),
    },

    "confusion_matrix": {

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),
    },
}


with open(
    RESULTS_DIR
    / "final_metrics.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        final_metrics,
        file,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# SALVAR MELHORES MODELOS
# ============================================================

classifier_path = (
    MODELS_DIR
    / f"classifier_{best_classifier_name}.joblib"
)


regressor_path = (
    MODELS_DIR
    / f"regressor_{best_regressor_name}.joblib"
)


joblib.dump(
    best_classifier,
    classifier_path,
)


joblib.dump(
    best_regressor,
    regressor_path,
)


# ============================================================
# RESUMO FINAL
# ============================================================

print("\n" + "=" * 70)
print("TREINAMENTO FINALIZADO")
print("=" * 70)

print(
    f"\nClassificador: "
    f"{best_classifier_name}"
)

print(
    f"Threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"F1 teste: "
    f"{test_class_metrics['f1']:.4f}"
)

print(
    f"\nRegressor: "
    f"{best_regressor_name}"
)

print(
    f"MAE teste: "
    f"{test_regression_metrics['mae']:.4f} mm"
)

print(
    f"RMSE teste: "
    f"{test_regression_metrics['rmse']:.4f} mm"
)

print(
    f"R² teste: "
    f"{test_regression_metrics['r2']:.4f}"
)

print(
    f"\nEnd-to-end MAE: "
    f"{end_to_end_mae:.6f} mm"
)

print(
    f"End-to-end RMSE: "
    f"{end_to_end_rmse:.6f} mm"
)

print(
    f"End-to-end R²: "
    f"{end_to_end_r2:.6f}"
)

print(
    f"\nModelos salvos em:"
)

print(
    classifier_path
)

print(
    regressor_path
)

print(
    "\nResultados salvos em:"
)

print(
    RESULTS_DIR
)

print("\n" + "=" * 70)