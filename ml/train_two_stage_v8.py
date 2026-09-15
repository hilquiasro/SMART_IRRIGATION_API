from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
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
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "irrigation_dataset.csv"

RESULTS_DIR = BASE_DIR / "results_8"
MODELS_DIR = BASE_DIR / "models_8"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "soil_moisture_percent",
    "temperature_c",
    "air_humidity_percent",
    "wind_speed_m_s",
    "solar_radiation_kwh_m2",
    "precipitation_mm",
    "kc",
    "et0",
    "etc",
    "soil_moisture_lag_1h",
    "soil_moisture_lag_3h",
    "soil_moisture_lag_6h",
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
]

CATEGORICAL_FEATURES = [
    "crop_stage",
]


# ============================================================
# MODELOS
# ============================================================

CLASSIFIERS = {
    "logistic_regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),

    "decision_tree": DecisionTreeClassifier(
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),

    "extra_trees": ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),

    # Mais adequado para dataset grande que GradientBoosting tradicional
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=42,
    ),

    "mlp": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=2048,
        learning_rate="adaptive",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
    ),
}


REGRESSORS = {
    "linear_regression": LinearRegression(),

    "decision_tree": DecisionTreeRegressor(
        min_samples_leaf=5,
        random_state=42,
    ),

    "random_forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    ),

    "extra_trees": ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    ),

    "hist_gradient_boosting": HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        loss="squared_error",
        random_state=42,
    ),

    "mlp": MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=1024,
        learning_rate="adaptive",
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
    ),
}


# ============================================================
# PREPROCESSAMENTO
# ============================================================

def build_preprocessor(scale_numeric=True):

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================
# PIPELINE
# ============================================================

def build_pipeline(model, scale_numeric=True):

    return Pipeline([
        (
            "preprocessor",
            build_preprocessor(scale_numeric=scale_numeric),
        ),
        ("model", model),
    ])


# ============================================================
# THRESHOLD
# ============================================================

def find_best_threshold(y_true, probabilities):

    thresholds = np.arange(
        0.05,
        0.951,
        0.01,
    )

    best_threshold = 0.5
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

    return float(best_threshold), float(best_f1)


# ============================================================
# CARREGAMENTO
# ============================================================

print("=" * 78)
print("TREINAMENTO TWO-STAGE — DATASET 3 — RODADA 8")
print("=" * 78)

print(f"Arquivo: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

print(f"Registros carregados: {len(df):,}")
print(f"Colunas: {len(df.columns)}")


# ============================================================
# TARGET
# ============================================================

df["irrigation_event"] = (
    df["irrigation_depth_mm"] > 0
).astype(int)


# ============================================================
# DIVISÃO TEMPORAL
# ============================================================

train_df = df[
    df["timestamp"].dt.year <= 2023
].copy()

validation_df = df[
    df["timestamp"].dt.year == 2024
].copy()

test_df = df[
    df["timestamp"].dt.year == 2025
].copy()


print()
print("=" * 78)
print("DIVISÃO TEMPORAL")
print("=" * 78)

print(
    f"Treinamento  2020–2023: "
    f"{len(train_df):,} registros"
)

print(
    f"Validação    2024:     "
    f"{len(validation_df):,} registros"
)

print(
    f"Teste        2025:     "
    f"{len(test_df):,} registros"
)

print()

print(
    f"Eventos treinamento: "
    f"{train_df['irrigation_event'].sum():,}"
)

print(
    f"Eventos validação:   "
    f"{validation_df['irrigation_event'].sum():,}"
)

print(
    f"Eventos teste:       "
    f"{test_df['irrigation_event'].sum():,}"
)


# ============================================================
# MATRIZES
# ============================================================

X_train = train_df[
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
]

X_val = validation_df[
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
]

X_test = test_df[
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
]

y_train_cls = train_df[
    "irrigation_event"
]

y_val_cls = validation_df[
    "irrigation_event"
]

y_test_cls = test_df[
    "irrigation_event"
]


# ============================================================
# 1. CLASSIFICAÇÃO
# ============================================================

print()
print("=" * 78)
print("1. CLASSIFICAÇÃO — IRRIGAR OU NÃO")
print("=" * 78)

classification_results = []

classifier_objects = {}

for name, model in CLASSIFIERS.items():

    print()
    print(f"Treinando: {name}")

    # HistGradientBoosting trabalha melhor com dados numéricos
    # densos. A pipeline abaixo mantém a representação consistente.
    scale_numeric = name in [
        "logistic_regression",
        "mlp",
    ]

    pipeline = build_pipeline(
        model,
        scale_numeric=scale_numeric,
    )

    pipeline.fit(
        X_train,
        y_train_cls,
    )

    probabilities = pipeline.predict_proba(
        X_val
    )[:, 1]

    threshold, best_f1 = find_best_threshold(
        y_val_cls,
        probabilities,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val_cls,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val_cls,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_val_cls,
        probabilities,
    )

    accuracy = accuracy_score(
        y_val_cls,
        predictions,
    )

    print(
        f"  threshold = {threshold:.2f}"
    )

    print(
        f"  accuracy  = {accuracy:.4f}"
    )

    print(
        f"  precision = {precision:.4f}"
    )

    print(
        f"  recall    = {recall:.4f}"
    )

    print(
        f"  F1        = {best_f1:.4f}"
    )

    print(
        f"  ROC-AUC   = {roc_auc:.4f}"
    )

    classification_results.append({
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": best_f1,
        "roc_auc": roc_auc,
    })

    classifier_objects[name] = pipeline


classification_results_df = pd.DataFrame(
    classification_results
).sort_values(
    "f1",
    ascending=False,
)

classification_results_df.to_csv(
    RESULTS_DIR / "classification_validation.csv",
    index=False,
)


best_classifier_name = (
    classification_results_df.iloc[0]["model"]
)

best_threshold = float(
    classification_results_df.iloc[0]["threshold"]
)

best_classifier = classifier_objects[
    best_classifier_name
]


print()
print("-" * 78)

print(
    f"Melhor classificador na validação: "
    f"{best_classifier_name}"
)

print(
    f"F1 de validação: "
    f"{classification_results_df.iloc[0]['f1']:.4f}"
)

print(
    f"Threshold escolhido: "
    f"{best_threshold:.2f}"
)


# ============================================================
# 2. REGRESSÃO
# ============================================================

print()
print("=" * 78)
print("2. REGRESSÃO — QUANTOS MM IRRIGAR")
print("=" * 78)

train_events = train_df[
    train_df["irrigation_event"] == 1
].copy()

validation_events = validation_df[
    validation_df["irrigation_event"] == 1
].copy()

X_train_reg = train_events[
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
]

y_train_reg = train_events[
    "irrigation_depth_mm"
]

X_val_reg = validation_events[
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
]

y_val_reg = validation_events[
    "irrigation_depth_mm"
]


print(
    f"Eventos para treinamento: "
    f"{len(train_events):,}"
)

print(
    f"Eventos para validação:   "
    f"{len(validation_events):,}"
)


regression_results = []

regressor_objects = {}

for name, model in REGRESSORS.items():

    print()
    print(f"Treinando: {name}")

    scale_numeric = name == "mlp"

    pipeline = build_pipeline(
        model,
        scale_numeric=scale_numeric,
    )

    pipeline.fit(
        X_train_reg,
        y_train_reg,
    )

    predictions = pipeline.predict(
        X_val_reg
    )

    predictions = np.clip(
        predictions,
        0,
        None,
    )

    mae = mean_absolute_error(
        y_val_reg,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val_reg,
            predictions,
        )
    )

    r2 = r2_score(
        y_val_reg,
        predictions,
    )

    print(
        f"  MAE  = {mae:.4f} mm"
    )

    print(
        f"  RMSE = {rmse:.4f} mm"
    )

    print(
        f"  R²   = {r2:.4f}"
    )

    regression_results.append({
        "model": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    })

    regressor_objects[name] = pipeline


regression_results_df = pd.DataFrame(
    regression_results
).sort_values(
    "mae",
    ascending=True,
)

regression_results_df.to_csv(
    RESULTS_DIR / "regression_validation.csv",
    index=False,
)


best_regressor_name = (
    regression_results_df.iloc[0]["model"]
)

best_regressor = regressor_objects[
    best_regressor_name
]


print()
print("-" * 78)

print(
    f"Melhor regressor na validação: "
    f"{best_regressor_name}"
)

print(
    f"MAE de validação: "
    f"{regression_results_df.iloc[0]['mae']:.4f} mm"
)


# ============================================================
# 3. TESTE FINAL — 2025
# ============================================================

print()
print("=" * 78)
print("3. AVALIAÇÃO FINAL — TESTE 2025")
print("=" * 78)


# ------------------------------------------------------------
# CLASSIFICAÇÃO
# ------------------------------------------------------------

test_probabilities = (
    best_classifier.predict_proba(
        X_test
    )[:, 1]
)

test_class_predictions = (
    test_probabilities >= best_threshold
).astype(int)


test_accuracy = accuracy_score(
    y_test_cls,
    test_class_predictions,
)

test_precision = precision_score(
    y_test_cls,
    test_class_predictions,
    zero_division=0,
)

test_recall = recall_score(
    y_test_cls,
    test_class_predictions,
    zero_division=0,
)

test_f1 = f1_score(
    y_test_cls,
    test_class_predictions,
    zero_division=0,
)

test_roc_auc = roc_auc_score(
    y_test_cls,
    test_probabilities,
)


print()
print("CLASSIFICAÇÃO")

print(
    f"Modelo:     {best_classifier_name}"
)

print(
    f"Threshold:  {best_threshold:.2f}"
)

print(
    f"Accuracy:   {test_accuracy:.4f}"
)

print(
    f"Precision:  {test_precision:.4f}"
)

print(
    f"Recall:     {test_recall:.4f}"
)

print(
    f"F1:         {test_f1:.4f}"
)

print(
    f"ROC-AUC:    {test_roc_auc:.4f}"
)


# ------------------------------------------------------------
# REGRESSÃO
# ------------------------------------------------------------

test_events = test_df[
    test_df["irrigation_event"] == 1
].copy()

X_test_reg = test_events[
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
]

y_test_reg = test_events[
    "irrigation_depth_mm"
]

regression_predictions = (
    best_regressor.predict(
        X_test_reg
    )
)

regression_predictions = np.clip(
    regression_predictions,
    0,
    None,
)


test_reg_mae = mean_absolute_error(
    y_test_reg,
    regression_predictions,
)

test_reg_rmse = np.sqrt(
    mean_squared_error(
        y_test_reg,
        regression_predictions,
    )
)

test_reg_r2 = r2_score(
    y_test_reg,
    regression_predictions,
)


print()
print("REGRESSÃO — SOMENTE EVENTOS REAIS")

print(
    f"Modelo: {best_regressor_name}"
)

print(
    f"MAE:  {test_reg_mae:.4f} mm"
)

print(
    f"RMSE: {test_reg_rmse:.4f} mm"
)

print(
    f"R²:   {test_reg_r2:.4f}"
)


# ============================================================
# END-TO-END
# ============================================================

end_to_end_predictions = np.zeros(
    len(test_df),
    dtype=float,
)

positive_mask = (
    test_class_predictions == 1
)

if positive_mask.any():

    X_positive = X_test.loc[
        positive_mask
    ]

    predicted_depths = (
        best_regressor.predict(
            X_positive
        )
    )

    predicted_depths = np.clip(
        predicted_depths,
        0,
        None,
    )

    end_to_end_predictions[
        positive_mask
    ] = predicted_depths


y_test_end = test_df[
    "irrigation_depth_mm"
].to_numpy()


end_mae = mean_absolute_error(
    y_test_end,
    end_to_end_predictions,
)

end_rmse = np.sqrt(
    mean_squared_error(
        y_test_end,
        end_to_end_predictions,
    )
)

end_r2 = r2_score(
    y_test_end,
    end_to_end_predictions,
)


real_water = float(
    y_test_end.sum()
)

predicted_water = float(
    end_to_end_predictions.sum()
)

water_difference = (
    predicted_water - real_water
)

water_difference_percent = (
    water_difference / real_water * 100
    if real_water > 0
    else 0
)


real_events = int(
    (y_test_end > 0).sum()
)

predicted_events = int(
    (end_to_end_predictions > 0).sum()
)


print()
print("END-TO-END")

print(
    f"MAE:                 {end_mae:.6f} mm"
)

print(
    f"RMSE:                {end_rmse:.6f} mm"
)

print(
    f"R²:                  {end_r2:.6f}"
)

print(
    f"Água real:            {real_water:.4f} mm"
)

print(
    f"Água prevista:        {predicted_water:.4f} mm"
)

print(
    f"Diferença:            {water_difference:+.4f} mm"
)

print(
    f"Diferença percentual: "
    f"{water_difference_percent:+.2f}%"
)

print(
    f"Eventos reais:        {real_events:,}"
)

print(
    f"Eventos previstos:    {predicted_events:,}"
)


# ============================================================
# PREDIÇÕES COMPLETAS
# ============================================================

predictions_df = test_df[
    [
        "id",
        "timestamp",
        "crop_stage",
        "soil_moisture_percent",
        "precipitation_mm",
        "rain_6h",
        "rain_24h",
        "etc",
        "etc_6h",
        "etc_24h",
        "irrigation_depth_mm",
    ]
].copy()

predictions_df[
    "irrigation_probability"
] = test_probabilities

predictions_df[
    "predicted_irrigation_event"
] = test_class_predictions

predictions_df[
    "predicted_irrigation_depth_mm"
] = end_to_end_predictions

predictions_df[
    "absolute_error_mm"
] = np.abs(
    predictions_df[
        "irrigation_depth_mm"
    ]
    -
    predictions_df[
        "predicted_irrigation_depth_mm"
    ]
)

predictions_df.to_csv(
    RESULTS_DIR / "predictions_test_2025.csv",
    index=False,
)


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

cm = confusion_matrix(
    y_test_cls,
    test_class_predictions,
)

confusion_df = pd.DataFrame(
    cm,
    index=[
        "real_nao_irrigar",
        "real_irrigar",
    ],
    columns=[
        "prev_nao_irrigar",
        "prev_irrigar",
    ],
)

confusion_df.to_csv(
    RESULTS_DIR / "confusion_matrix_test.csv"
)


# ============================================================
# IMPORTÂNCIA DAS FEATURES — EXTRA TREES
# ============================================================

print()
print("=" * 78)
print("4. IMPORTÂNCIA DAS FEATURES — EXTRA TREES")
print("=" * 78)

extra_trees_classifier = (
    classifier_objects[
        "extra_trees"
    ]
)

try:

    preprocessor = (
        extra_trees_classifier
        .named_steps["preprocessor"]
    )

    model = (
        extra_trees_classifier
        .named_steps["model"]
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = (
        model.feature_importances_
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(
        "importance",
        ascending=False,
    )

    importance_df.to_csv(
        RESULTS_DIR /
        "feature_importance_extra_trees.csv",
        index=False,
    )

    print()
    print(
        "Top 20 features:"
    )

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

except Exception as exc:

    print(
        "Não foi possível calcular "
        f"importância: {exc}"
    )


# ============================================================
# MÉTRICAS FINAIS
# ============================================================

final_metrics = {

    "dataset": "irrigation_dataset_3.csv",

    "training_period": "2020-2023",

    "validation_period": "2024",

    "test_period": "2025",

    "best_classifier": best_classifier_name,

    "classification_threshold": best_threshold,

    "classification_test": {
        "accuracy": float(test_accuracy),
        "precision": float(test_precision),
        "recall": float(test_recall),
        "f1": float(test_f1),
        "roc_auc": float(test_roc_auc),
    },

    "best_regressor": best_regressor_name,

    "regression_test": {
        "mae_mm": float(test_reg_mae),
        "rmse_mm": float(test_reg_rmse),
        "r2": float(test_reg_r2),
    },

    "end_to_end_test": {
        "mae_mm": float(end_mae),
        "rmse_mm": float(end_rmse),
        "r2": float(end_r2),
        "real_water_mm": float(real_water),
        "predicted_water_mm": float(predicted_water),
        "difference_water_mm": float(water_difference),
        "difference_water_percent": float(
            water_difference_percent
        ),
        "real_events": real_events,
        "predicted_events": predicted_events,
    },

    "feature_policy": {
        "target_excluded": True,
        "storage_before_irrigation_excluded": True,
        "storage_after_irrigation_excluded": True,
        "dynamic_lower_bound_excluded": True,
        "scenario_id_excluded": True,
        "storage_capacity_excluded": True,
        "future_features_used": False,
    },

    "notes": [
        "Dataset 3 was kept unchanged.",
        "The temporal split was preserved.",
        "GradientBoosting tradicional was replaced by HistGradientBoosting.",
        "The positive class was handled with class weights only in classifiers.",
        "Regression models were trained only on positive irrigation events.",
        "The 2025 test set was not used for model selection.",
    ],
}


with open(
    RESULTS_DIR / "final_metrics.json",
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
# CONFIGURAÇÃO DOS MODELOS
# ============================================================

model_config = {

    "dataset": str(DATA_PATH),

    "results_dir": str(RESULTS_DIR),

    "models_dir": str(MODELS_DIR),

    "numeric_features": NUMERIC_FEATURES,

    "categorical_features": CATEGORICAL_FEATURES,

    "best_classifier": best_classifier_name,

    "classification_threshold": best_threshold,

    "best_regressor": best_regressor_name,

    "excluded_features": [
        "irrigation_depth_mm",
        "irrigation_event",
        "storage_before_irrigation_mm",
        "storage_after_irrigation_mm",
        "dynamic_lower_bound_percent",
        "scenario_id",
        "storage_capacity_mm",
    ],

    "split": {
        "train": "2020-2023",
        "validation": "2024",
        "test": "2025",
    },
}


with open(
    MODELS_DIR / "model_config.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        model_config,
        file,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# SALVAR MODELOS
# ============================================================

print()
print("=" * 78)
print("SALVANDO MODELOS")
print("=" * 78)

for name, model in classifier_objects.items():

    output_path = (
        MODELS_DIR /
        f"classifier_{name}.joblib"
    )

    joblib.dump(
        model,
        output_path,
    )

    print(
        f"Classificador salvo: "
        f"{output_path.name}"
    )


for name, model in regressor_objects.items():

    output_path = (
        MODELS_DIR /
        f"regressor_{name}.joblib"
    )

    joblib.dump(
        model,
        output_path,
    )

    print(
        f"Regressor salvo: "
        f"{output_path.name}"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 78)
print("TREINAMENTO CONCLUÍDO")
print("=" * 78)

print(
    f"Melhor classificador: "
    f"{best_classifier_name}"
)

print(
    f"Threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Melhor regressor: "
    f"{best_regressor_name}"
)

print()

print(
    f"Modelos: {MODELS_DIR}"
)

print(
    f"Resultados: {RESULTS_DIR}"
)

print()

print("Arquivos principais:")

print(
    "  - classification_validation.csv"
)

print(
    "  - regression_validation.csv"
)

print(
    "  - predictions_test_2025.csv"
)

print(
    "  - confusion_matrix_test.csv"
)

print(
    "  - feature_importance_extra_trees.csv"
)

print(
    "  - final_metrics.json"
)

print(
    "  - model_config.json"
)

print("=" * 78)