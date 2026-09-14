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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
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

RESULTS_DIR = BASE_DIR / "results_6"
MODELS_DIR = BASE_DIR / "models_6"

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
# PREPROCESSAMENTO
# ============================================================

def build_preprocessor(scale_numeric=False):

    if scale_numeric:

        numeric_transformer = Pipeline(
            [
                ("scaler", StandardScaler()),
            ]
        )

    else:

        numeric_transformer = "passthrough"

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


# ============================================================
# MODELOS
# ============================================================

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
                    __import__(
                        "sklearn.linear_model",
                        fromlist=["LinearRegression"],
                    ).LinearRegression(),
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
    }


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def classification_metrics(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "predicted_events": int(
            predictions.sum()
        ),
        "real_events": int(
            y_true.sum()
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

df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# ============================================================
# VALIDAÇÃO
# ============================================================

required_columns = (
    FEATURES
    + [
        TARGET,
        "timestamp",
    ]
)

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:

    raise ValueError(
        f"Colunas ausentes: {missing}"
    )

if df[required_columns].isna().any().any():

    raise ValueError(
        "Existem valores ausentes."
    )


# ============================================================
# CLASSE BINÁRIA
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
    "Eventos treinamento:",
    int(
        train_df["irrigation_event"].sum()
    ),
)

print(
    "Eventos validação:",
    int(
        validation_df["irrigation_event"].sum()
    ),
)

print(
    "Eventos teste:",
    int(
        test_df["irrigation_event"].sum()
    ),
)


# ============================================================
# MATRIZES DE CLASSIFICAÇÃO
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df[
    "irrigation_event"
]

X_val = validation_df[FEATURES]
y_val = validation_df[
    "irrigation_event"
]

X_test = test_df[FEATURES]
y_test = test_df[
    "irrigation_event"
]


# ============================================================
# ESTÁGIO 1
# CLASSIFICAÇÃO
# ============================================================

print()
print("=" * 70)
print("ESTÁGIO 1 — CLASSIFICAÇÃO")
print("=" * 70)

classifiers = (
    build_classifier_models()
)

classifier_models = {}

classifier_summary = []

for name, model in classifiers.items():

    print(
        f"\nTreinando: {name}"
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = (
        model.predict_proba(X_val)[:, 1]
    )

    metrics = classification_metrics(
        y_val,
        probabilities,
        0.50,
    )

    metrics["model"] = name

    classifier_summary.append(
        metrics
    )

    classifier_models[name] = model

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1:        "
        f"{metrics['f1']:.4f}"
    )


classifier_summary_df = pd.DataFrame(
    classifier_summary
)

classifier_summary_df.to_csv(
    RESULTS_DIR
    / "classifiers_validation.csv",
    index=False,
)


# ============================================================
# MELHOR CLASSIFICADOR
# ============================================================

best_classifier_name = (
    classifier_summary_df
    .sort_values(
        "f1",
        ascending=False,
    )
    .iloc[0]["model"]
)

best_classifier = (
    classifier_models[
        best_classifier_name
    ]
)


print()
print(
    "Melhor classificador:",
    best_classifier_name,
)


# ============================================================
# PROBABILIDADES DA VALIDAÇÃO
# ============================================================

val_probabilities = (
    best_classifier
    .predict_proba(X_val)[:, 1]
)


# ============================================================
# ANÁLISE COMPLETA DE THRESHOLDS
# ============================================================

print()
print("=" * 70)
print("ANÁLISE DE THRESHOLD")
print("=" * 70)

thresholds = np.arange(
    0.10,
    0.96,
    0.01,
)

threshold_results = []

for threshold in thresholds:

    metrics = classification_metrics(
        y_val,
        val_probabilities,
        float(threshold),
    )

    threshold_results.append(
        metrics
    )


threshold_df = pd.DataFrame(
    threshold_results
)


# ============================================================
# ESTÁGIO 2
# REGRESSÃO
# ============================================================

print()
print("=" * 70)
print("ESTÁGIO 2 — REGRESSÃO")
print("=" * 70)

train_events = train_df[
    train_df[TARGET] > 0
].copy()

validation_events = validation_df[
    validation_df[TARGET] > 0
].copy()

test_events = test_df[
    test_df[TARGET] > 0
].copy()


X_train_reg = train_events[
    FEATURES
]

y_train_reg = train_events[
    TARGET
]

X_val_reg = validation_events[
    FEATURES
]

y_val_reg = validation_events[
    TARGET
]


regressors = (
    build_regressor_models()
)

regressor_models = {}

regressor_summary = []

for name, model in regressors.items():

    print(
        f"\nTreinando: {name}"
    )

    model.fit(
        X_train_reg,
        y_train_reg,
    )

    predictions = model.predict(
        X_val_reg
    )

    predictions = np.clip(
        predictions,
        0.0,
        None,
    )

    metrics = {
        "model": name,

        "mae": mean_absolute_error(
            y_val_reg,
            predictions,
        ),

        "rmse": np.sqrt(
            mean_squared_error(
                y_val_reg,
                predictions,
            )
        ),

        "r2": r2_score(
            y_val_reg,
            predictions,
        ),
    }

    regressor_summary.append(
        metrics
    )

    regressor_models[name] = model

    print(
        f"MAE:  "
        f"{metrics['mae']:.4f}"
    )

    print(
        f"RMSE: "
        f"{metrics['rmse']:.4f}"
    )

    print(
        f"R²:   "
        f"{metrics['r2']:.4f}"
    )


regressor_summary_df = pd.DataFrame(
    regressor_summary
)

regressor_summary_df.to_csv(
    RESULTS_DIR
    / "regressors_validation.csv",
    index=False,
)


# ============================================================
# MELHOR REGRESSOR
# ============================================================

best_regressor_name = (
    regressor_summary_df
    .sort_values(
        "mae",
        ascending=True,
    )
    .iloc[0]["model"]
)

best_regressor = (
    regressor_models[
        best_regressor_name
    ]
)


print()
print(
    "Melhor regressor:",
    best_regressor_name,
)


# ============================================================
# FUNÇÃO PARA GERAR SISTEMA END-TO-END
# ============================================================

def predict_system(
    X,
    probabilities,
    regressor,
    threshold,
):

    event_prediction = (
        probabilities >= threshold
    )

    depth_prediction = np.zeros(
        len(X)
    )

    indices = np.where(
        event_prediction
    )[0]

    if len(indices) > 0:

        depth_prediction[
            indices
        ] = np.clip(
            regressor.predict(
                X.iloc[indices]
            ),
            0.0,
            None,
        )

    return (
        event_prediction,
        depth_prediction,
    )


# ============================================================
# ANÁLISE DE THRESHOLD END-TO-END
# ============================================================

print()
print("=" * 70)
print("ANÁLISE END-TO-END DOS THRESHOLDS")
print("=" * 70)

y_val_depth = (
    validation_df[TARGET]
    .to_numpy()
)

end_to_end_thresholds = []

for threshold in thresholds:

    event_prediction, depth_prediction = (
        predict_system(
            X_val,
            val_probabilities,
            best_regressor,
            float(threshold),
        )
    )

    classification = (
        classification_metrics(
            y_val,
            val_probabilities,
            float(threshold),
        )
    )

    overall_mae = (
        mean_absolute_error(
            y_val_depth,
            depth_prediction,
        )
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            y_val_depth,
            depth_prediction,
        )
    )

    overall_r2 = r2_score(
        y_val_depth,
        depth_prediction,
    )

    real_water = (
        y_val_depth.sum()
    )

    predicted_water = (
        depth_prediction.sum()
    )

    water_difference = (
        predicted_water
        - real_water
    )

    end_to_end_thresholds.append(
        {
            **classification,

            "overall_mae":
                overall_mae,

            "overall_rmse":
                overall_rmse,

            "overall_r2":
                overall_r2,

            "real_water_mm":
                real_water,

            "predicted_water_mm":
                predicted_water,

            "water_difference_mm":
                water_difference,
        }
    )


end_to_end_df = pd.DataFrame(
    end_to_end_thresholds
)


# ============================================================
# SALVAR ANÁLISE
# ============================================================

threshold_df.to_csv(
    RESULTS_DIR
    / "threshold_analysis_validation.csv",
    index=False,
)

end_to_end_df.to_csv(
    RESULTS_DIR
    / "end_to_end_threshold_analysis_validation.csv",
    index=False,
)


# ============================================================
# THRESHOLDS IMPORTANTES
# ============================================================

# Melhor F1
best_f1_row = (
    end_to_end_df
    .sort_values(
        "f1",
        ascending=False,
    )
    .iloc[0]
)


# Maior recall mantendo precision >= 80%
precision_80 = end_to_end_df[
    end_to_end_df["precision"] >= 0.80
]

if not precision_80.empty:

    best_precision_80_row = (
        precision_80
        .sort_values(
            "recall",
            ascending=False,
        )
        .iloc[0]
    )

else:

    best_precision_80_row = None


# Maior recall mantendo precision >= 75%
precision_75 = end_to_end_df[
    end_to_end_df["precision"] >= 0.75
]

if not precision_75.empty:

    best_precision_75_row = (
        precision_75
        .sort_values(
            "recall",
            ascending=False,
        )
        .iloc[0]
    )

else:

    best_precision_75_row = None


# Maior precision
best_precision_row = (
    end_to_end_df
    .sort_values(
        "precision",
        ascending=False,
    )
    .iloc[0]
)


# ============================================================
# EXIBIÇÃO
# ============================================================

print()
print("=" * 70)
print("MELHORES THRESHOLDS — VALIDAÇÃO 2024")
print("=" * 70)

print()
print("Melhor F1:")

print(
    f"  threshold = "
    f"{best_f1_row['threshold']:.2f}"
)

print(
    f"  precision = "
    f"{best_f1_row['precision']:.4f}"
)

print(
    f"  recall    = "
    f"{best_f1_row['recall']:.4f}"
)

print(
    f"  F1        = "
    f"{best_f1_row['f1']:.4f}"
)

print(
    f"  água      = "
    f"{best_f1_row['predicted_water_mm']:.3f} mm"
)


if best_precision_80_row is not None:

    print()
    print(
        "Maior recall com "
        "precision >= 80%:"
    )

    print(
        f"  threshold = "
        f"{best_precision_80_row['threshold']:.2f}"
    )

    print(
        f"  precision = "
        f"{best_precision_80_row['precision']:.4f}"
    )

    print(
        f"  recall    = "
        f"{best_precision_80_row['recall']:.4f}"
    )

    print(
        f"  F1        = "
        f"{best_precision_80_row['f1']:.4f}"
    )

    print(
        f"  água      = "
        f"{best_precision_80_row['predicted_water_mm']:.3f} mm"
    )


if best_precision_75_row is not None:

    print()
    print(
        "Maior recall com "
        "precision >= 75%:"
    )

    print(
        f"  threshold = "
        f"{best_precision_75_row['threshold']:.2f}"
    )

    print(
        f"  precision = "
        f"{best_precision_75_row['precision']:.4f}"
    )

    print(
        f"  recall    = "
        f"{best_precision_75_row['recall']:.4f}"
    )

    print(
        f"  F1        = "
        f"{best_precision_75_row['f1']:.4f}"
    )

    print(
        f"  água      = "
        f"{best_precision_75_row['predicted_water_mm']:.3f} mm"
    )


print()
print("Maior precision:")

print(
    f"  threshold = "
    f"{best_precision_row['threshold']:.2f}"
)

print(
    f"  precision = "
    f"{best_precision_row['precision']:.4f}"
)

print(
    f"  recall    = "
    f"{best_precision_row['recall']:.4f}"
)

print(
    f"  F1        = "
    f"{best_precision_row['f1']:.4f}"
)


# ============================================================
# THRESHOLD FINAL
# ============================================================

if best_precision_80_row is not None:

    selected_threshold = float(
        best_precision_80_row[
            "threshold"
        ]
    )

else:

    selected_threshold = float(
        best_f1_row[
            "threshold"
        ]
    )


print()
print(
    "=" * 70
)

print(
    f"THRESHOLD SELECIONADO: "
    f"{selected_threshold:.2f}"
)

print(
    "=" * 70
)


# ============================================================
# TESTE FINAL 2025
# ============================================================

print()
print("=" * 70)
print("TESTE FINAL — 2025")
print("=" * 70)

test_probabilities = (
    best_classifier
    .predict_proba(X_test)[:, 1]
)

test_event_prediction, test_depth_prediction = (
    predict_system(
        X_test,
        test_probabilities,
        best_regressor,
        selected_threshold,
    )
)


y_test_depth = (
    test_df[TARGET]
    .to_numpy()
)


# ============================================================
# MÉTRICAS DE CLASSIFICAÇÃO
# ============================================================

test_precision = precision_score(
    y_test,
    test_event_prediction,
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    test_event_prediction,
    zero_division=0,
)

test_f1 = f1_score(
    y_test,
    test_event_prediction,
    zero_division=0,
)

test_roc_auc = roc_auc_score(
    y_test,
    test_probabilities,
)


tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_event_prediction,
    labels=[0, 1],
).ravel()


# ============================================================
# MÉTRICAS DE REGRESSÃO
# ============================================================

test_overall_mae = (
    mean_absolute_error(
        y_test_depth,
        test_depth_prediction,
    )
)

test_overall_rmse = np.sqrt(
    mean_squared_error(
        y_test_depth,
        test_depth_prediction,
    )
)

test_overall_r2 = r2_score(
    y_test_depth,
    test_depth_prediction,
)


# ============================================================
# REGRESSÃO APENAS NOS EVENTOS REAIS
# ============================================================

event_mask = (
    y_test_depth > 0
)

if event_mask.any():

    event_mae = mean_absolute_error(
        y_test_depth[event_mask],
        test_depth_prediction[event_mask],
    )

    event_rmse = np.sqrt(
        mean_squared_error(
            y_test_depth[event_mask],
            test_depth_prediction[event_mask],
        )
    )

    event_r2 = r2_score(
        y_test_depth[event_mask],
        test_depth_prediction[event_mask],
    )

else:

    event_mae = np.nan
    event_rmse = np.nan
    event_r2 = np.nan


# ============================================================
# ÁGUA
# ============================================================

real_water = (
    y_test_depth.sum()
)

predicted_water = (
    test_depth_prediction.sum()
)

water_difference = (
    predicted_water
    - real_water
)

water_difference_percent = (
    water_difference
    / real_water
    * 100
)


# ============================================================
# RESULTADO
# ============================================================

test_results = {

    "classifier":
        best_classifier_name,

    "regressor":
        best_regressor_name,

    "threshold":
        selected_threshold,

    "overall_mae":
        test_overall_mae,

    "overall_rmse":
        test_overall_rmse,

    "overall_r2":
        test_overall_r2,

    "event_mae":
        event_mae,

    "event_rmse":
        event_rmse,

    "event_r2":
        event_r2,

    "precision":
        test_precision,

    "recall":
        test_recall,

    "f1":
        test_f1,

    "roc_auc":
        test_roc_auc,

    "real_events":
        int(y_test.sum()),

    "predicted_events":
        int(test_event_prediction.sum()),

    "true_positive":
        int(tp),

    "false_positive":
        int(fp),

    "true_negative":
        int(tn),

    "false_negative":
        int(fn),

    "real_water_mm":
        real_water,

    "predicted_water_mm":
        predicted_water,

    "water_difference_mm":
        water_difference,

    "water_difference_percent":
        water_difference_percent,
}


pd.DataFrame(
    [test_results]
).to_csv(
    RESULTS_DIR
    / "final_test_2025.csv",
    index=False,
)


# ============================================================
# PREDIÇÕES 2025
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
    test_event_prediction.astype(int)
)


predictions_df[
    "predicted_irrigation_depth_mm"
] = (
    test_depth_prediction
)


predictions_df.to_csv(
    RESULTS_DIR
    / "predictions_2025.csv",
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
# IMPRESSÃO FINAL
# ============================================================

print()
print("=" * 70)
print("RESULTADO FINAL — EXPERIMENTO 6")
print("=" * 70)

print()
print(
    f"Classificador: "
    f"{best_classifier_name}"
)

print(
    f"Regressor:     "
    f"{best_regressor_name}"
)

print(
    f"Threshold:     "
    f"{selected_threshold:.2f}"
)

print()

print(
    f"Precision:     "
    f"{test_precision:.6f}"
)

print(
    f"Recall:        "
    f"{test_recall:.6f}"
)

print(
    f"F1:            "
    f"{test_f1:.6f}"
)

print(
    f"ROC-AUC:       "
    f"{test_roc_auc:.6f}"
)

print()

print(
    f"Overall MAE:   "
    f"{test_overall_mae:.6f} mm"
)

print(
    f"Overall RMSE:  "
    f"{test_overall_rmse:.6f} mm"
)

print(
    f"Overall R²:    "
    f"{test_overall_r2:.6f}"
)

print()

print(
    f"Event MAE:     "
    f"{event_mae:.6f} mm"
)

print(
    f"Event RMSE:    "
    f"{event_rmse:.6f} mm"
)

print(
    f"Event R²:      "
    f"{event_r2:.6f}"
)

print()

print(
    f"Eventos reais: "
    f"{int(y_test.sum())}"
)

print(
    f"Eventos previstos: "
    f"{int(test_event_prediction.sum())}"
)

print()

print(
    f"TP: {tp}"
)

print(
    f"FP: {fp}"
)

print(
    f"FN: {fn}"
)

print()

print(
    f"Água real:     "
    f"{real_water:.3f} mm"
)

print(
    f"Água prevista: "
    f"{predicted_water:.3f} mm"
)

print(
    f"Diferença:     "
    f"{water_difference:.3f} mm"
)

print(
    f"Diferença %:   "
    f"{water_difference_percent:.2f}%"
)

print()

print(
    f"Resultados: "
    f"{RESULTS_DIR}"
)

print(
    f"Modelos: "
    f"{MODELS_DIR}"
)

print("=" * 70)