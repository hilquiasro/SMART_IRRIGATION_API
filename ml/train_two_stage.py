from pathlib import Path

import joblib
import matplotlib.pyplot as plt
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
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# ============================================================
# CONFIGURAÇÃO
# ============================================================

RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "irrigation_dataset.csv"

RESULTS_DIR = BASE_DIR / "results_3"
MODELS_DIR = BASE_DIR / "models_3"

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

EVENT_THRESHOLD = 0.1


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def create_preprocessor():

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

    return ColumnTransformer(
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


def regression_metrics(y_true, y_pred):

    y_pred = np.clip(
        y_pred,
        0,
        None
    )

    return {
        "mae": mean_absolute_error(
            y_true,
            y_pred
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        ),
        "r2": r2_score(
            y_true,
            y_pred
        ),
    }


def classification_metrics(y_true, y_pred, y_prob):

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),
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
    }

    if len(np.unique(y_true)) == 2:

        metrics["roc_auc"] = roc_auc_score(
            y_true,
            y_prob
        )

    else:

        metrics["roc_auc"] = np.nan

    metrics["real_events"] = int(
        np.sum(y_true == 1)
    )

    metrics["predicted_events"] = int(
        np.sum(y_pred == 1)
    )

    return metrics


# ============================================================
# INÍCIO
# ============================================================

print("=" * 70)
print("EXPERIMENTO 3 - MODELO EM DUAS ETAPAS")
print("=" * 70)

print()
print(f"Dataset: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print(
    f"Registros carregados: {len(df):,}"
)


# ============================================================
# VALIDAÇÃO
# ============================================================

required_columns = [
    *FEATURES,
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


df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# ============================================================
# VARIÁVEL DE CLASSIFICAÇÃO
# ============================================================

df["irrigation_event"] = (
    df[TARGET] > 0
).astype(int)


print()
print("Distribuição do alvo:")

event_counts = (
    df["irrigation_event"]
    .value_counts()
    .sort_index()
)

print(
    f"Sem irrigação: "
    f"{event_counts.get(0, 0):,}"
)

print(
    f"Com irrigação: "
    f"{event_counts.get(1, 0):,}"
)


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
print(
    f"Treino: {len(train_df):,} registros"
)

print(
    f"Teste : {len(test_df):,} registros"
)


# ============================================================
# MATRIZES
# ============================================================

X_train = train_df[FEATURES]
X_test = test_df[FEATURES]

y_train_event = train_df[
    "irrigation_event"
]

y_test_event = test_df[
    "irrigation_event"
]


# ============================================================
# ETAPA 1
# CLASSIFICAÇÃO
# ============================================================

print()
print("=" * 70)
print("ETAPA 1 - CLASSIFICAÇÃO")
print("=" * 70)


classifiers = {

    "logistic_regression":
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),

    "decision_tree":
        DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),

    "random_forest":
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),

    "extra_trees":
        ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),

    "gradient_boosting":
        GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_SEED
        ),
}


classifier_results = []
classifier_predictions = {}
classifier_probabilities = {}


for name, model in classifiers.items():

    print()
    print("-" * 70)
    print(f"Treinando classificador: {name}")
    print("-" * 70)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "model",
                model
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train_event
    )

    y_pred = pipeline.predict(
        X_test
    )

    y_prob = pipeline.predict_proba(
        X_test
    )[:, 1]

    metrics = classification_metrics(
        y_test_event.values,
        y_pred,
        y_prob
    )

    classifier_results.append(
        {
            "model": name,
            **metrics
        }
    )

    classifier_predictions[name] = y_pred
    classifier_probabilities[name] = y_prob

    print(
        f"Accuracy : "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall   : "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F1       : "
        f"{metrics['f1']:.6f}"
    )

    print(
        f"ROC-AUC  : "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"Eventos reais: "
        f"{metrics['real_events']}"
    )

    print(
        f"Eventos previstos: "
        f"{metrics['predicted_events']}"
    )


classifier_results_df = pd.DataFrame(
    classifier_results
)

classifier_results_df = (
    classifier_results_df
    .sort_values(
        "f1",
        ascending=False
    )
    .reset_index(drop=True)
)


print()
print("=" * 70)
print("COMPARAÇÃO DOS CLASSIFICADORES")
print("=" * 70)

print(
    classifier_results_df.to_string(
        index=False
    )
)


classifier_results_df.to_csv(
    RESULTS_DIR /
    "classifier_comparison.csv",
    index=False
)


# ============================================================
# MELHOR CLASSIFICADOR
# ============================================================

best_classifier_row = (
    classifier_results_df.iloc[0]
)

best_classifier_name = (
    best_classifier_row["model"]
)

best_classifier_prediction = (
    classifier_predictions[
        best_classifier_name
    ]
)

best_classifier_probability = (
    classifier_probabilities[
        best_classifier_name
    ]
)


print()
print(
    f"Melhor classificador pelo F1: "
    f"{best_classifier_name}"
)


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

cm = confusion_matrix(
    y_test_event,
    best_classifier_prediction
)

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title(
    f"Matriz de confusão - "
    f"{best_classifier_name}"
)

plt.xlabel(
    "Classe prevista"
)

plt.ylabel(
    "Classe real"
)

plt.xticks(
    [0, 1],
    ["Não irrigar", "Irrigar"]
)

plt.yticks(
    [0, 1],
    ["Não irrigar", "Irrigar"]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# ETAPA 2
# REGRESSÃO DOS EVENTOS
# ============================================================

print()
print("=" * 70)
print("ETAPA 2 - REGRESSÃO DOS EVENTOS")
print("=" * 70)


train_events = train_df[
    train_df[TARGET] > 0
].copy()

test_events = test_df[
    test_df[TARGET] > 0
].copy()


print()
print(
    f"Eventos disponíveis para treinamento: "
    f"{len(train_events):,}"
)

print(
    f"Eventos disponíveis para teste: "
    f"{len(test_events):,}"
)


X_train_events = train_events[
    FEATURES
]

y_train_events = train_events[
    TARGET
]

X_test_events = test_events[
    FEATURES
]

y_test_events = test_events[
    TARGET
]


regressors = {

    "linear_regression":
        LinearRegression(),

    "decision_tree":
        DecisionTreeRegressor(
            max_depth=12,
            min_samples_leaf=2,
            random_state=RANDOM_SEED
        ),

    "random_forest":
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),

    "extra_trees":
        ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),

    "gradient_boosting":
        GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_SEED,
            loss="squared_error"
        ),
}


regressor_results = []
regressor_predictions = {}


for name, model in regressors.items():

    print()
    print("-" * 70)
    print(
        f"Treinando regressor: {name}"
    )
    print("-" * 70)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "model",
                model
            ),
        ]
    )

    pipeline.fit(
        X_train_events,
        y_train_events
    )

    y_pred = pipeline.predict(
        X_test_events
    )

    y_pred = np.clip(
        y_pred,
        0,
        None
    )

    metrics = regression_metrics(
        y_test_events.values,
        y_pred
    )

    regressor_results.append(
        {
            "model": name,
            **metrics
        }
    )

    regressor_predictions[name] = y_pred

    print(
        f"MAE : {metrics['mae']:.6f}"
    )

    print(
        f"RMSE: {metrics['rmse']:.6f}"
    )

    print(
        f"R²  : {metrics['r2']:.6f}"
    )


regressor_results_df = pd.DataFrame(
    regressor_results
)

regressor_results_df = (
    regressor_results_df
    .sort_values(
        "mae"
    )
    .reset_index(drop=True)
)


print()
print("=" * 70)
print("COMPARAÇÃO DOS REGRESSORES")
print("=" * 70)

print(
    regressor_results_df.to_string(
        index=False
    )
)


regressor_results_df.to_csv(
    RESULTS_DIR /
    "regressor_comparison.csv",
    index=False
)


# ============================================================
# MELHOR REGRESSOR
# ============================================================

best_regressor_row = (
    regressor_results_df.iloc[0]
)

best_regressor_name = (
    best_regressor_row["model"]
)

best_regressor_prediction_events = (
    regressor_predictions[
        best_regressor_name
    ]
)


print()
print(
    f"Melhor regressor pelo MAE: "
    f"{best_regressor_name}"
)


# ============================================================
# AVALIAÇÃO DO SISTEMA COMPLETO
# ============================================================

print()
print("=" * 70)
print("AVALIAÇÃO DO SISTEMA EM DUAS ETAPAS")
print("=" * 70)


# ------------------------------------------------------------
# Recriar os modelos escolhidos
# ------------------------------------------------------------

best_classifier_model = classifiers[
    best_classifier_name
]

best_classifier_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "model",
            best_classifier_model
        ),
    ]
)

best_classifier_pipeline.fit(
    X_train,
    y_train_event
)


best_regressor_model = regressors[
    best_regressor_name
]

best_regressor_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "model",
            best_regressor_model
        ),
    ]
)

best_regressor_pipeline.fit(
    X_train_events,
    y_train_events
)


# ------------------------------------------------------------
# Classificação de todo o teste
# ------------------------------------------------------------

test_event_prediction = (
    best_classifier_pipeline.predict(
        X_test
    )
)


# ------------------------------------------------------------
# Regressão para todos os registros
# ------------------------------------------------------------

predicted_depth = np.zeros(
    len(test_df)
)


predicted_event_mask = (
    test_event_prediction == 1
)


if predicted_event_mask.any():

    predicted_depth[
        predicted_event_mask
    ] = best_regressor_pipeline.predict(
        X_test.loc[
            predicted_event_mask
        ]
    )


predicted_depth = np.clip(
    predicted_depth,
    0,
    None
)


y_test_depth = test_df[
    TARGET
].values


# ============================================================
# MÉTRICAS FINAIS
# ============================================================

overall_mae = mean_absolute_error(
    y_test_depth,
    predicted_depth
)

overall_rmse = np.sqrt(
    mean_squared_error(
        y_test_depth,
        predicted_depth
    )
)

overall_r2 = r2_score(
    y_test_depth,
    predicted_depth
)


real_event_mask = (
    y_test_depth > 0
)

real_event_depth = (
    y_test_depth[
        real_event_mask
    ]
)

predicted_for_real_events = (
    predicted_depth[
        real_event_mask
    ]
)


event_mae = mean_absolute_error(
    real_event_depth,
    predicted_for_real_events
)

event_rmse = np.sqrt(
    mean_squared_error(
        real_event_depth,
        predicted_for_real_events
    )
)

event_r2 = r2_score(
    real_event_depth,
    predicted_for_real_events
)


final_metrics = {

    "classifier": best_classifier_name,

    "regressor": best_regressor_name,

    "overall_mae": overall_mae,

    "overall_rmse": overall_rmse,

    "overall_r2": overall_r2,

    "event_mae": event_mae,

    "event_rmse": event_rmse,

    "event_r2": event_r2,

    "classification_precision":
        precision_score(
            y_test_event,
            test_event_prediction,
            zero_division=0
        ),

    "classification_recall":
        recall_score(
            y_test_event,
            test_event_prediction,
            zero_division=0
        ),

    "classification_f1":
        f1_score(
            y_test_event,
            test_event_prediction,
            zero_division=0
        ),

    "real_events":
        int(real_event_mask.sum()),

    "predicted_events":
        int(predicted_event_mask.sum()),
}


print()
print(
    f"Classificador: "
    f"{best_classifier_name}"
)

print(
    f"Regressor: "
    f"{best_regressor_name}"
)

print()
print(
    f"MAE geral       : "
    f"{overall_mae:.6f}"
)

print(
    f"RMSE geral      : "
    f"{overall_rmse:.6f}"
)

print(
    f"R² geral        : "
    f"{overall_r2:.6f}"
)

print()
print(
    f"MAE eventos     : "
    f"{event_mae:.6f}"
)

print(
    f"RMSE eventos    : "
    f"{event_rmse:.6f}"
)

print(
    f"R² eventos      : "
    f"{event_r2:.6f}"
)

print()
print(
    f"Precisão        : "
    f"{final_metrics['classification_precision']:.6f}"
)

print(
    f"Recall          : "
    f"{final_metrics['classification_recall']:.6f}"
)

print(
    f"F1              : "
    f"{final_metrics['classification_f1']:.6f}"
)

print(
    f"Eventos reais   : "
    f"{final_metrics['real_events']}"
)

print(
    f"Eventos previstos: "
    f"{final_metrics['predicted_events']}"
)


# ============================================================
# SALVAR MÉTRICAS FINAIS
# ============================================================

pd.DataFrame(
    [final_metrics]
).to_csv(
    RESULTS_DIR /
    "two_stage_final_metrics.csv",
    index=False
)


# ============================================================
# PREVISÕES
# ============================================================

predictions_df = test_df[
    [
        "timestamp",
        TARGET
    ]
].copy()


predictions_df[
    "actual_event"
] = y_test_event.values


predictions_df[
    "predicted_event"
] = test_event_prediction


predictions_df[
    "predicted_irrigation_depth_mm"
] = predicted_depth


predictions_df.to_csv(
    RESULTS_DIR /
    "two_stage_predictions.csv",
    index=False
)


# ============================================================
# GRÁFICO - REGRESSÃO DOS EVENTOS
# ============================================================

plt.figure(figsize=(9, 7))

plt.scatter(
    y_test_events,
    best_regressor_prediction_events,
    alpha=0.4,
    s=15
)

max_value = max(
    y_test_events.max(),
    best_regressor_prediction_events.max()
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
    "Regressão da profundidade nos eventos de irrigação"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "event_actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ============================================================
# GRÁFICO - SISTEMA COMPLETO
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test_depth,
    predicted_depth,
    alpha=0.25,
    s=10
)

max_value = max(
    y_test_depth.max(),
    predicted_depth.max()
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
    "Sistema em duas etapas - valores reais vs previstos"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "two_stage_actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ============================================================
# GRÁFICO - COMPARAÇÃO CLASSIFICADORES
# ============================================================

plt.figure(figsize=(10, 6))

ordered = classifier_results_df.sort_values(
    "f1"
)

plt.bar(
    ordered["model"],
    ordered["f1"]
)

plt.xlabel(
    "Modelo"
)

plt.ylabel(
    "F1"
)

plt.title(
    "Comparação dos classificadores"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "classifier_f1_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# GRÁFICO - COMPARAÇÃO REGRESSORES
# ============================================================

plt.figure(figsize=(10, 6))

ordered = regressor_results_df.sort_values(
    "mae"
)

plt.bar(
    ordered["model"],
    ordered["mae"]
)

plt.xlabel(
    "Modelo"
)

plt.ylabel(
    "MAE (mm)"
)

plt.title(
    "Comparação dos regressores nos eventos"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "regressor_mae_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# SALVAR MODELOS
# ============================================================

joblib.dump(
    best_classifier_pipeline,
    MODELS_DIR /
    "best_classifier.joblib"
)

joblib.dump(
    best_regressor_pipeline,
    MODELS_DIR /
    "best_regressor.joblib"
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("EXPERIMENTO 3 CONCLUÍDO")
print("=" * 70)

print()
print("Resultados:")
print(RESULTS_DIR)

print()
print("Modelos:")
print(MODELS_DIR)

print()
print("Arquivos principais:")

print(
    "- classifier_comparison.csv"
)

print(
    "- regressor_comparison.csv"
)

print(
    "- two_stage_final_metrics.csv"
)

print(
    "- two_stage_predictions.csv"
)

print(
    "- confusion_matrix.png"
)

print(
    "- event_actual_vs_predicted.png"
)

print(
    "- two_stage_actual_vs_predicted.png"
)