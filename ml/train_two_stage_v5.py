from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "irrigation_dataset.csv"

RESULTS_DIR = BASE_DIR / "results_5"
MODELS_DIR = BASE_DIR / "models_5"

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

TARGET = "irrigation_depth_mm"


# ============================================================
# FUNÇÕES
# ============================================================

def build_preprocessor(scale_numeric=False):
    """
    Cria o pré-processamento.

    Para modelos baseados em árvores, não precisamos
    normalizar as variáveis numéricas.

    Para modelos lineares e MLP, usamos StandardScaler.
    """

    numeric_steps = []

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    from sklearn.pipeline import Pipeline as SkPipeline

    numeric_transformer = (
        SkPipeline(numeric_steps)
        if numeric_steps
        else "passthrough"
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
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


def build_classifier_models():
    return {
        "logistic_regression": Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(
                        scale_numeric=True
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),

        "decision_tree": Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "model",
                    DecisionTreeClassifier(
                        class_weight="balanced",
                        max_depth=12,
                        min_samples_leaf=5,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),

        "random_forest": Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        min_samples_leaf=2,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "extra_trees": Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        min_samples_leaf=2,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "gradient_boosting": Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),

        "mlp": Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(
                        scale_numeric=True
                    ),
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
        ),
    }


def build_regressor_models():
    return {
        "linear_regression": Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(
                        scale_numeric=True
                    ),
                ),
                (
                    "model",
                    LinearRegression(),
                ),
            ]
        ),

        "decision_tree": Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=12,
                        min_samples_leaf=5,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),

        "random_forest": Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "extra_trees": Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "gradient_boosting": Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),

        "mlp": Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(
                        scale_numeric=True
                    ),
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
        ),
    }


def calculate_classification_metrics(
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "real_events": int(y_true.sum()),
        "predicted_events": int(
            predictions.sum()
        ),
    }


# ============================================================
# CARREGAMENTO
# ============================================================

print("Carregando dataset V4...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["timestamp"],
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ============================================================
# VALIDAÇÃO
# ============================================================

required_columns = FEATURES + [
    TARGET,
    "timestamp",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Colunas ausentes: {missing_columns}"
    )

if df[required_columns].isna().any().any():
    raise ValueError(
        "Existem valores ausentes."
    )


# ============================================================
# TARGET DE CLASSIFICAÇÃO
# ============================================================

df["irrigation_event"] = (
    df[TARGET] > 0
).astype(int)


# ============================================================
# DIVISÃO TEMPORAL
# ============================================================

train_df = df[
    df["timestamp"] < "2024-01-01"
].copy()

validation_df = df[
    (df["timestamp"] >= "2024-01-01")
    & (df["timestamp"] < "2025-01-01")
].copy()

test_df = df[
    df["timestamp"] >= "2025-01-01"
].copy()


print()
print("=" * 70)
print("DIVISÃO TEMPORAL")
print("=" * 70)

print(
    f"Treinamento: {len(train_df):,}"
)

print(
    f"Validação:   {len(validation_df):,}"
)

print(
    f"Teste:       {len(test_df):,}"
)

print()

print(
    "Eventos no treinamento:",
    int(train_df["irrigation_event"].sum()),
)

print(
    "Eventos na validação:",
    int(validation_df["irrigation_event"].sum()),
)

print(
    "Eventos no teste:",
    int(test_df["irrigation_event"].sum()),
)


# ============================================================
# MATRIZES
# ============================================================

X_train = train_df[FEATURES]
y_train_event = train_df[
    "irrigation_event"
]

X_val = validation_df[FEATURES]
y_val_event = validation_df[
    "irrigation_event"
]

X_test = test_df[FEATURES]
y_test_event = test_df[
    "irrigation_event"
]


# ============================================================
# 1. CLASSIFICADORES
# ============================================================

print()
print("=" * 70)
print("ESTÁGIO 1 — CLASSIFICAÇÃO")
print("=" * 70)

classifier_models = (
    build_classifier_models()
)

classifier_results = []

fitted_classifiers = {}

for name, model in classifier_models.items():

    print(
        f"\nTreinando classificador: {name}"
    )

    model.fit(
        X_train,
        y_train_event,
    )

    validation_probabilities = (
        model.predict_proba(X_val)[:, 1]
    )

    # --------------------------------------------------------
    # Avaliação inicial em threshold 0.5
    # --------------------------------------------------------

    metrics_05 = calculate_classification_metrics(
        y_val_event,
        validation_probabilities,
        0.50,
    )

    metrics_05["model"] = name

    classifier_results.append(
        metrics_05
    )

    fitted_classifiers[name] = model

    print(
        f"  F1 (0.50): "
        f"{metrics_05['f1']:.4f}"
    )

    print(
        f"  Precision: "
        f"{metrics_05['precision']:.4f}"
    )

    print(
        f"  Recall: "
        f"{metrics_05['recall']:.4f}"
    )

    print(
        f"  ROC-AUC: "
        f"{metrics_05['roc_auc']:.4f}"
    )


classifier_results_df = pd.DataFrame(
    classifier_results
)

classifier_results_df.to_csv(
    RESULTS_DIR
    / "classifier_validation_threshold_050.csv",
    index=False,
)


# ============================================================
# SELEÇÃO DO MELHOR CLASSIFICADOR
# ============================================================

best_classifier_name = (
    classifier_results_df
    .sort_values(
        "f1",
        ascending=False,
    )
    .iloc[0]["model"]
)

best_classifier = fitted_classifiers[
    best_classifier_name
]


print()
print(
    "Melhor classificador inicial:",
    best_classifier_name,
)


# ============================================================
# OTIMIZAÇÃO DO THRESHOLD
# ============================================================

print()
print("=" * 70)
print("OTIMIZAÇÃO DO THRESHOLD")
print("=" * 70)

validation_probabilities = (
    best_classifier
    .predict_proba(X_val)[:, 1]
)

threshold_results = []

thresholds = np.arange(
    0.10,
    0.96,
    0.05,
)

for threshold in thresholds:

    metrics = calculate_classification_metrics(
        y_val_event,
        validation_probabilities,
        float(threshold),
    )

    threshold_results.append(
        metrics
    )


threshold_df = pd.DataFrame(
    threshold_results
)

threshold_df.to_csv(
    RESULTS_DIR
    / "threshold_validation.csv",
    index=False,
)


# ------------------------------------------------------------
# Melhor threshold por F1
# ------------------------------------------------------------

best_threshold_f1 = (
    threshold_df
    .sort_values(
        "f1",
        ascending=False,
    )
    .iloc[0]["threshold"]
)


# ------------------------------------------------------------
# Melhor threshold com precision >= 80%
# ------------------------------------------------------------

precision_candidates = (
    threshold_df[
        threshold_df["precision"] >= 0.80
    ]
)

if not precision_candidates.empty:

    best_threshold_precision = (
        precision_candidates
        .sort_values(
            "recall",
            ascending=False,
        )
        .iloc[0]["threshold"]
    )

else:

    best_threshold_precision = np.nan


print()
print(
    f"Melhor threshold por F1: "
    f"{best_threshold_f1:.2f}"
)

if not np.isnan(
    best_threshold_precision
):

    print(
        "Melhor threshold com "
        "precision >= 80%:",
        f"{best_threshold_precision:.2f}",
    )

else:

    print(
        "Nenhum threshold atingiu "
        "precision >= 80% na validação."
    )


# ============================================================
# ESCOLHA FINAL DO THRESHOLD
# ============================================================

if not np.isnan(
    best_threshold_precision
):

    selected_threshold = (
        best_threshold_precision
    )

else:

    selected_threshold = (
        best_threshold_f1
    )


print(
    f"\nThreshold selecionado: "
    f"{selected_threshold:.2f}"
)


# ============================================================
# 2. REGRESSORES
# ============================================================

print()
print("=" * 70)
print("ESTÁGIO 2 — REGRESSÃO")
print("=" * 70)

train_event_df = train_df[
    train_df[TARGET] > 0
].copy()

validation_event_df = validation_df[
    validation_df[TARGET] > 0
].copy()

test_event_df = test_df[
    test_df[TARGET] > 0
].copy()


X_train_reg = train_event_df[
    FEATURES
]

y_train_reg = train_event_df[
    TARGET
]

X_val_reg = validation_event_df[
    FEATURES
]

y_val_reg = validation_event_df[
    TARGET
]

X_test_reg = test_event_df[
    FEATURES
]

y_test_reg = test_event_df[
    TARGET
]


print()
print(
    "Eventos usados no treinamento:",
    len(train_event_df),
)

print(
    "Eventos usados na validação:",
    len(validation_event_df),
)

print(
    "Eventos usados no teste:",
    len(test_event_df),
)


regressor_models = (
    build_regressor_models()
)

regressor_results = []

fitted_regressors = {}

for name, model in regressor_models.items():

    print(
        f"\nTreinando regressor: {name}"
    )

    model.fit(
        X_train_reg,
        y_train_reg,
    )

    validation_predictions = (
        model.predict(X_val_reg)
    )

    validation_predictions = np.clip(
        validation_predictions,
        0.0,
        None,
    )

    metrics = {
        "model": name,
        "mae": mean_absolute_error(
            y_val_reg,
            validation_predictions,
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                y_val_reg,
                validation_predictions,
            )
        ),
        "r2": r2_score(
            y_val_reg,
            validation_predictions,
        ),
    }

    regressor_results.append(
        metrics
    )

    fitted_regressors[name] = model

    print(
        f"  MAE:  {metrics['mae']:.4f}"
    )

    print(
        f"  RMSE: {metrics['rmse']:.4f}"
    )

    print(
        f"  R²:   {metrics['r2']:.4f}"
    )


regressor_results_df = pd.DataFrame(
    regressor_results
)

regressor_results_df.to_csv(
    RESULTS_DIR
    / "regressor_validation.csv",
    index=False,
)


# ============================================================
# MELHOR REGRESSOR
# ============================================================

best_regressor_name = (
    regressor_results_df
    .sort_values(
        "mae",
        ascending=True,
    )
    .iloc[0]["model"]
)

best_regressor = fitted_regressors[
    best_regressor_name
]


print()
print(
    "Melhor regressor:",
    best_regressor_name,
)


# ============================================================
# 3. MODELO END-TO-END NA VALIDAÇÃO
# ============================================================

print()
print("=" * 70)
print("SISTEMA END-TO-END — VALIDAÇÃO")
print("=" * 70)

validation_probabilities = (
    best_classifier
    .predict_proba(X_val)[:, 1]
)

validation_event_predictions = (
    validation_probabilities
    >= selected_threshold
)

validation_depth_predictions = (
    np.zeros(len(X_val))
)

validation_event_indices = np.where(
    validation_event_predictions
)[0]

if len(validation_event_indices) > 0:

    validation_depth_predictions[
        validation_event_indices
    ] = np.clip(
        best_regressor.predict(
            X_val.iloc[
                validation_event_indices
            ]
        ),
        0.0,
        None,
    )


validation_y_true = (
    validation_df[TARGET]
    .to_numpy()
)

validation_overall_mae = (
    mean_absolute_error(
        validation_y_true,
        validation_depth_predictions,
    )
)

validation_overall_rmse = np.sqrt(
    mean_squared_error(
        validation_y_true,
        validation_depth_predictions,
    )
)

validation_overall_r2 = r2_score(
    validation_y_true,
    validation_depth_predictions,
)

validation_event_mask = (
    validation_y_true > 0
)

if validation_event_mask.any():

    validation_event_mae = (
        mean_absolute_error(
            validation_y_true[
                validation_event_mask
            ],
            validation_depth_predictions[
                validation_event_mask
            ],
        )
    )

    validation_event_rmse = np.sqrt(
        mean_squared_error(
            validation_y_true[
                validation_event_mask
            ],
            validation_depth_predictions[
                validation_event_mask
            ],
        )
    )

    validation_event_r2 = r2_score(
        validation_y_true[
            validation_event_mask
        ],
        validation_depth_predictions[
            validation_event_mask
        ],
    )

else:

    validation_event_mae = np.nan
    validation_event_rmse = np.nan
    validation_event_r2 = np.nan


validation_precision = precision_score(
    y_val_event,
    validation_event_predictions,
    zero_division=0,
)

validation_recall = recall_score(
    y_val_event,
    validation_event_predictions,
    zero_division=0,
)

validation_f1 = f1_score(
    y_val_event,
    validation_event_predictions,
    zero_division=0,
)


validation_metrics = {
    "classifier": best_classifier_name,
    "regressor": best_regressor_name,
    "threshold": selected_threshold,

    "overall_mae": validation_overall_mae,
    "overall_rmse": validation_overall_rmse,
    "overall_r2": validation_overall_r2,

    "event_mae": validation_event_mae,
    "event_rmse": validation_event_rmse,
    "event_r2": validation_event_r2,

    "precision": validation_precision,
    "recall": validation_recall,
    "f1": validation_f1,

    "real_events": int(
        y_val_event.sum()
    ),

    "predicted_events": int(
        validation_event_predictions.sum()
    ),
}


pd.DataFrame(
    [validation_metrics]
).to_csv(
    RESULTS_DIR
    / "end_to_end_validation.csv",
    index=False,
)


for key, value in validation_metrics.items():
    print(
        f"{key:20s}: {value}"
    )


# ============================================================
# 4. TESTE FINAL
# ============================================================

print()
print("=" * 70)
print("SISTEMA END-TO-END — TESTE FINAL 2025")
print("=" * 70)

test_probabilities = (
    best_classifier
    .predict_proba(X_test)[:, 1]
)

test_event_predictions = (
    test_probabilities
    >= selected_threshold
)

test_depth_predictions = (
    np.zeros(len(X_test))
)

test_event_indices = np.where(
    test_event_predictions
)[0]

if len(test_event_indices) > 0:

    test_depth_predictions[
        test_event_indices
    ] = np.clip(
        best_regressor.predict(
            X_test.iloc[
                test_event_indices
            ]
        ),
        0.0,
        None,
    )


test_y_true = (
    test_df[TARGET]
    .to_numpy()
)

test_overall_mae = (
    mean_absolute_error(
        test_y_true,
        test_depth_predictions,
    )
)

test_overall_rmse = np.sqrt(
    mean_squared_error(
        test_y_true,
        test_depth_predictions,
    )
)

test_overall_r2 = r2_score(
    test_y_true,
    test_depth_predictions,
)

test_event_mask = (
    test_y_true > 0
)

if test_event_mask.any():

    test_event_mae = (
        mean_absolute_error(
            test_y_true[
                test_event_mask
            ],
            test_depth_predictions[
                test_event_mask
            ],
        )
    )

    test_event_rmse = np.sqrt(
        mean_squared_error(
            test_y_true[
                test_event_mask
            ],
            test_depth_predictions[
                test_event_mask
            ],
        )
    )

    test_event_r2 = r2_score(
        test_y_true[
            test_event_mask
        ],
        test_depth_predictions[
            test_event_mask
        ],
    )

else:

    test_event_mae = np.nan
    test_event_rmse = np.nan
    test_event_r2 = np.nan


test_precision = precision_score(
    y_test_event,
    test_event_predictions,
    zero_division=0,
)

test_recall = recall_score(
    y_test_event,
    test_event_predictions,
    zero_division=0,
)

test_f1 = f1_score(
    y_test_event,
    test_event_predictions,
    zero_division=0,
)

test_roc_auc = roc_auc_score(
    y_test_event,
    test_probabilities,
)


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test_event,
    test_event_predictions,
).ravel()


# ============================================================
# ÁGUA REAL × PREVISTA
# ============================================================

real_water = (
    test_y_true.sum()
)

predicted_water = (
    test_depth_predictions.sum()
)

water_difference = (
    predicted_water
    - real_water
)


test_metrics = {
    "classifier": best_classifier_name,
    "regressor": best_regressor_name,
    "threshold": selected_threshold,

    "overall_mae": test_overall_mae,
    "overall_rmse": test_overall_rmse,
    "overall_r2": test_overall_r2,

    "event_mae": test_event_mae,
    "event_rmse": test_event_rmse,
    "event_r2": test_event_r2,

    "precision": test_precision,
    "recall": test_recall,
    "f1": test_f1,
    "roc_auc": test_roc_auc,

    "real_events": int(
        y_test_event.sum()
    ),

    "predicted_events": int(
        test_event_predictions.sum()
    ),

    "true_positives": int(tp),
    "false_positives": int(fp),
    "true_negatives": int(tn),
    "false_negatives": int(fn),

    "real_water_mm": real_water,
    "predicted_water_mm": predicted_water,
    "water_difference_mm": water_difference,
}


pd.DataFrame(
    [test_metrics]
).to_csv(
    RESULTS_DIR
    / "end_to_end_test_2025.csv",
    index=False,
)


# ============================================================
# PREDIÇÕES
# ============================================================

predictions_df = test_df[
    [
        "id",
        "timestamp",
        "soil_moisture_percent",
        "temperature_c",
        "air_humidity_percent",
        "solar_radiation_kwh_m2",
        "precipitation_mm",
        "crop_stage",
        TARGET,
    ]
].copy()

predictions_df[
    "irrigation_probability"
] = test_probabilities

predictions_df[
    "predicted_irrigation_event"
] = (
    test_event_predictions.astype(int)
)

predictions_df[
    "predicted_irrigation_depth_mm"
] = test_depth_predictions


predictions_df.to_csv(
    RESULTS_DIR
    / "test_predictions_2025.csv",
    index=False,
)


# ============================================================
# SALVAR MODELOS
# ============================================================

joblib.dump(
    best_classifier,
    MODELS_DIR
    / "best_classifier.joblib",
)

joblib.dump(
    best_regressor,
    MODELS_DIR
    / "best_regressor.joblib",
)


# ============================================================
# RESULTADO FINAL
# ============================================================

print()
print("=" * 70)
print("RESULTADO FINAL — TESTE 2025")
print("=" * 70)

print()
print(
    f"Classificador: {best_classifier_name}"
)

print(
    f"Regressor:     {best_regressor_name}"
)

print(
    f"Threshold:     {selected_threshold:.2f}"
)

print()

print(
    f"Overall MAE:   {test_overall_mae:.6f}"
)

print(
    f"Overall RMSE:  {test_overall_rmse:.6f}"
)

print(
    f"Overall R²:    {test_overall_r2:.6f}"
)

print()

print(
    f"Event MAE:     {test_event_mae:.6f}"
)

print(
    f"Event RMSE:    {test_event_rmse:.6f}"
)

print(
    f"Event R²:      {test_event_r2:.6f}"
)

print()

print(
    f"Precision:     {test_precision:.6f}"
)

print(
    f"Recall:        {test_recall:.6f}"
)

print(
    f"F1:            {test_f1:.6f}"
)

print(
    f"ROC-AUC:       {test_roc_auc:.6f}"
)

print()

print(
    f"Eventos reais:     "
    f"{int(y_test_event.sum())}"
)

print(
    f"Eventos previstos: "
    f"{int(test_event_predictions.sum())}"
)

print()

print(
    f"Verdadeiros positivos: {tp}"
)

print(
    f"Falsos positivos:      {fp}"
)

print(
    f"Falsos negativos:      {fn}"
)

print()

print(
    f"Água real:      "
    f"{real_water:.3f} mm"
)

print(
    f"Água prevista:  "
    f"{predicted_water:.3f} mm"
)

print(
    f"Diferença:      "
    f"{water_difference:.3f} mm"
)

print()
print(
    f"Resultados salvos em: "
    f"{RESULTS_DIR}"
)

print(
    f"Modelos salvos em: "
    f"{MODELS_DIR}"
)

print("=" * 70)