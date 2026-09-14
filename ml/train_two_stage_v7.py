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
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
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
from sklearn.utils.class_weight import compute_sample_weight


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "irrigation_dataset_3.csv"

MODELS_DIR = BASE_DIR / "models_7"
RESULTS_DIR = BASE_DIR / "results_7"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FEATURES
# =============================================================================
#
# Estas são as variáveis que podem estar disponíveis no momento da previsão.
#
# NÃO incluir:
# - irrigation_depth_mm
# - storage_before_irrigation_mm
# - storage_after_irrigation_mm
# - dynamic_lower_bound_percent
# - scenario_id
#
# storage_capacity_mm também fica fora nesta primeira versão.
#
# Se alguma feature abaixo não existir no CSV, o script irá informar e parar.
# =============================================================================

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

TARGET = "irrigation_depth_mm"

ID_COLUMNS = [
    "id",
    "timestamp",
    "scenario_id",
]


# =============================================================================
# MODELOS
# =============================================================================

CLASSIFIERS = {
    "logistic_regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),

    "decision_tree": DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    ),

    "extra_trees": ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    ),

    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_SEED,
    ),

    "mlp": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=RANDOM_SEED,
    ),
}


REGRESSORS = {
    "linear_regression": LinearRegression(),

    "decision_tree": DecisionTreeRegressor(
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
    ),

    "random_forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    ),

    "extra_trees": ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        loss="huber",
        random_state=RANDOM_SEED,
    ),

    "mlp": MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=RANDOM_SEED,
    ),
}


# =============================================================================
# PREPROCESSAMENTO
# =============================================================================

def build_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(model):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


# =============================================================================
# LEITURA
# =============================================================================

def load_dataset():
    print("=" * 78)
    print("TREINAMENTO TWO-STAGE — DATASET 3")
    print("=" * 78)

    print(f"Arquivo: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Registros carregados: {len(df):,}")
    print(f"Colunas: {len(df.columns)}")

    required_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [TARGET]
        + ["timestamp"]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        print("\n❌ Colunas obrigatórias ausentes:")
        for column in missing:
            print(f"   - {column}")

        print("\nColunas disponíveis:")
        for column in df.columns:
            print(f"   - {column}")

        raise ValueError(
            "O dataset não possui todas as colunas esperadas."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# =============================================================================
# DIVISÃO TEMPORAL
# =============================================================================

def temporal_split(df):
    print("\n" + "=" * 78)
    print("DIVISÃO TEMPORAL")
    print("=" * 78)

    train = df[df["timestamp"].dt.year <= 2023].copy()

    validation = df[
        df["timestamp"].dt.year == 2024
    ].copy()

    test = df[
        df["timestamp"].dt.year == 2025
    ].copy()

    if len(train) == 0:
        raise ValueError("Conjunto de treinamento vazio.")

    if len(validation) == 0:
        raise ValueError("Conjunto de validação vazio.")

    if len(test) == 0:
        raise ValueError("Conjunto de teste vazio.")

    print(
        f"Treinamento  2020–2023: {len(train):,} registros"
    )
    print(
        f"Validação    2024:     {len(validation):,} registros"
    )
    print(
        f"Teste        2025:     {len(test):,} registros"
    )

    print("\nEventos:")

    print(
        f"Treinamento: "
        f"{(train[TARGET] > 0).sum():,}"
    )

    print(
        f"Validação:   "
        f"{(validation[TARGET] > 0).sum():,}"
    )

    print(
        f"Teste:       "
        f"{(test[TARGET] > 0).sum():,}"
    )

    return train, validation, test


# =============================================================================
# MATRIZES
# =============================================================================

def get_xy(df):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = (df[TARGET] > 0).astype(int)

    return X, y


def get_regression_xy(df):
    events = df[df[TARGET] > 0].copy()

    X = events[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ].copy()

    y = events[TARGET].astype(float)

    return X, y, events


# =============================================================================
# CLASSIFICAÇÃO
# =============================================================================

def evaluate_classifier(y_true, probability, threshold):
    prediction = (
        probability >= threshold
    ).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, prediction)
        ),
        "precision": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
    }

    try:
        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probability,
            )
        )
    except ValueError:
        metrics["roc_auc"] = np.nan

    return metrics


def find_best_threshold(y_true, probability):
    thresholds = np.arange(
        0.05,
        0.951,
        0.01,
    )

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:
        prediction = (
            probability >= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_true,
            prediction,
            zero_division=0,
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)

    return best_threshold


def train_classifiers(
    train,
    validation,
):
    print("\n" + "=" * 78)
    print("1. CLASSIFICAÇÃO — IRRIGAR OU NÃO")
    print("=" * 78)

    X_train, y_train = get_xy(train)
    X_val, y_val = get_xy(validation)

    results = []
    trained_models = {}

    for name, model in CLASSIFIERS.items():
        print(f"\nTreinando: {name}")

        pipeline = build_pipeline(model)

        fit_kwargs = {}

        # Para o MLP, usamos pesos de amostra porque ele não
        # possui class_weight.
        if name == "mlp":
            sample_weight = compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )

            fit_kwargs["model__sample_weight"] = sample_weight

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=ConvergenceWarning,
            )

            try:
                pipeline.fit(
                    X_train,
                    y_train,
                    **fit_kwargs,
                )
            except TypeError:
                # Compatibilidade com versões do sklearn em que
                # MLPClassifier não aceita sample_weight.
                if name == "mlp":
                    print(
                        "  Aviso: versão do scikit-learn "
                        "não aceita sample_weight no MLP."
                    )

                    pipeline.fit(
                        X_train,
                        y_train,
                    )
                else:
                    raise

        probability = pipeline.predict_proba(
            X_val
        )[:, 1]

        threshold = find_best_threshold(
            y_val,
            probability,
        )

        metrics = evaluate_classifier(
            y_val,
            probability,
            threshold,
        )

        metrics["model"] = name

        results.append(metrics)
        trained_models[name] = pipeline

        print(
            f"  threshold = {metrics['threshold']:.2f}"
        )
        print(
            f"  precision = {metrics['precision']:.4f}"
        )
        print(
            f"  recall    = {metrics['recall']:.4f}"
        )
        print(
            f"  F1        = {metrics['f1']:.4f}"
        )
        print(
            f"  ROC-AUC   = {metrics['roc_auc']:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "f1",
        ascending=False,
    ).reset_index(drop=True)

    results_df.to_csv(
        RESULTS_DIR / "classification_validation.csv",
        index=False,
    )

    best_name = results_df.iloc[0]["model"]
    best_threshold = float(
        results_df.iloc[0]["threshold"]
    )

    print("\n" + "-" * 78)
    print(
        f"Melhor classificador na validação: "
        f"{best_name}"
    )
    print(
        f"F1 de validação: "
        f"{results_df.iloc[0]['f1']:.4f}"
    )
    print(
        f"Threshold escolhido: "
        f"{best_threshold:.2f}"
    )

    return (
        results_df,
        trained_models,
        best_name,
        best_threshold,
    )


# =============================================================================
# REGRESSÃO
# =============================================================================

def evaluate_regressor(y_true, prediction):
    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                prediction,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    prediction,
                )
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                prediction,
            )
        ),
    }


def train_regressors(
    train,
    validation,
):
    print("\n" + "=" * 78)
    print("2. REGRESSÃO — QUANTOS MM IRRIGAR")
    print("=" * 78)

    X_train, y_train, train_events = (
        get_regression_xy(train)
    )

    X_val, y_val, val_events = (
        get_regression_xy(validation)
    )

    print(
        f"Eventos para treinamento: "
        f"{len(train_events):,}"
    )

    print(
        f"Eventos para validação:   "
        f"{len(val_events):,}"
    )

    results = []
    trained_models = {}

    for name, model in REGRESSORS.items():
        print(f"\nTreinando: {name}")

        pipeline = build_pipeline(model)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=ConvergenceWarning,
            )

            pipeline.fit(
                X_train,
                y_train,
            )

        prediction = pipeline.predict(
            X_val
        )

        prediction = np.maximum(
            prediction,
            0.0,
        )

        metrics = evaluate_regressor(
            y_val,
            prediction,
        )

        metrics["model"] = name

        results.append(metrics)
        trained_models[name] = pipeline

        print(
            f"  MAE  = {metrics['mae']:.4f} mm"
        )
        print(
            f"  RMSE = {metrics['rmse']:.4f} mm"
        )
        print(
            f"  R²   = {metrics['r2']:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "mae",
        ascending=True,
    ).reset_index(drop=True)

    results_df.to_csv(
        RESULTS_DIR / "regression_validation.csv",
        index=False,
    )

    best_name = results_df.iloc[0]["model"]

    print("\n" + "-" * 78)
    print(
        f"Melhor regressor na validação: "
        f"{best_name}"
    )
    print(
        f"MAE de validação: "
        f"{results_df.iloc[0]['mae']:.4f} mm"
    )

    return (
        results_df,
        trained_models,
        best_name,
    )


# =============================================================================
# AVALIAÇÃO FINAL — TESTE 2025
# =============================================================================

def evaluate_final_model(
    classifier,
    classifier_name,
    threshold,
    regressor,
    regressor_name,
    test,
):
    print("\n" + "=" * 78)
    print("3. AVALIAÇÃO FINAL — TESTE 2025")
    print("=" * 78)

    X_test = test[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ].copy()

    y_depth = test[TARGET].astype(float)
    y_event = (y_depth > 0).astype(int)

    # -------------------------------------------------------------------------
    # CLASSIFICADOR
    # -------------------------------------------------------------------------

    probability = classifier.predict_proba(
        X_test
    )[:, 1]

    predicted_event = (
        probability >= threshold
    ).astype(int)

    classification_metrics = {
        "classifier": classifier_name,
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                y_event,
                predicted_event,
            )
        ),
        "precision": float(
            precision_score(
                y_event,
                predicted_event,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_event,
                predicted_event,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_event,
                predicted_event,
                zero_division=0,
            )
        ),
    }

    try:
        classification_metrics["roc_auc"] = float(
            roc_auc_score(
                y_event,
                probability,
            )
        )
    except ValueError:
        classification_metrics["roc_auc"] = np.nan

    print("\nCLASSIFICAÇÃO")
    print(
        f"Modelo:     {classifier_name}"
    )
    print(
        f"Threshold:  {threshold:.2f}"
    )
    print(
        f"Accuracy:   "
        f"{classification_metrics['accuracy']:.4f}"
    )
    print(
        f"Precision:  "
        f"{classification_metrics['precision']:.4f}"
    )
    print(
        f"Recall:     "
        f"{classification_metrics['recall']:.4f}"
    )
    print(
        f"F1:         "
        f"{classification_metrics['f1']:.4f}"
    )
    print(
        f"ROC-AUC:    "
        f"{classification_metrics['roc_auc']:.4f}"
    )

    # -------------------------------------------------------------------------
    # REGRESSOR
    # -------------------------------------------------------------------------

    true_event_mask = y_event == 1

    regression_metrics = {}

    if true_event_mask.sum() > 0:
        X_test_events = X_test[
            true_event_mask
        ]

        y_test_events = y_depth[
            true_event_mask
        ]

        predicted_depth_events = (
            regressor.predict(
                X_test_events
            )
        )

        predicted_depth_events = np.maximum(
            predicted_depth_events,
            0.0,
        )

        regression_metrics = evaluate_regressor(
            y_test_events,
            predicted_depth_events,
        )

    print("\nREGRESSÃO — SOMENTE EVENTOS REAIS")
    print(
        f"Modelo: {regressor_name}"
    )
    print(
        f"MAE:  {regression_metrics.get('mae', np.nan):.4f} mm"
    )
    print(
        f"RMSE: {regression_metrics.get('rmse', np.nan):.4f} mm"
    )
    print(
        f"R²:   {regression_metrics.get('r2', np.nan):.4f}"
    )

    # -------------------------------------------------------------------------
    # END-TO-END
    # -------------------------------------------------------------------------
    #
    # O sistema final só irriga quando o classificador diz "sim".
    # Caso diga "não", profundidade prevista = 0.
    # -------------------------------------------------------------------------

    predicted_depth = np.zeros(
        len(test),
        dtype=float,
    )

    predicted_positive_mask = (
        predicted_event == 1
    )

    if predicted_positive_mask.sum() > 0:
        X_pred_events = X_test[
            predicted_positive_mask
        ]

        predicted_depth[predicted_positive_mask] = (
            regressor.predict(
                X_pred_events
            )
        )

        predicted_depth = np.maximum(
            predicted_depth,
            0.0,
        )

    # -------------------------------------------------------------------------
    # MÉTRICAS END-TO-END
    # -------------------------------------------------------------------------

    end_to_end_mae = mean_absolute_error(
        y_depth,
        predicted_depth,
    )

    end_to_end_rmse = np.sqrt(
        mean_squared_error(
            y_depth,
            predicted_depth,
        )
    )

    end_to_end_r2 = r2_score(
        y_depth,
        predicted_depth,
    )

    real_water = float(
        y_depth.sum()
    )

    predicted_water = float(
        predicted_depth.sum()
    )

    water_difference = (
        predicted_water - real_water
    )

    water_difference_percent = (
        100.0 * water_difference / real_water
        if real_water > 0
        else np.nan
    )

    end_to_end_metrics = {
        "classifier": classifier_name,
        "regressor": regressor_name,
        "threshold": float(threshold),

        "overall_mae_mm": float(
            end_to_end_mae
        ),
        "overall_rmse_mm": float(
            end_to_end_rmse
        ),
        "overall_r2": float(
            end_to_end_r2
        ),

        "real_water_mm": real_water,
        "predicted_water_mm": predicted_water,
        "water_difference_mm": float(
            water_difference
        ),
        "water_difference_percent": float(
            water_difference_percent
        ),

        "real_events": int(
            y_event.sum()
        ),
        "predicted_events": int(
            predicted_event.sum()
        ),
    }

    print("\nEND-TO-END")
    print(
        f"MAE:                 "
        f"{end_to_end_mae:.6f} mm"
    )
    print(
        f"RMSE:                "
        f"{end_to_end_rmse:.6f} mm"
    )
    print(
        f"R²:                  "
        f"{end_to_end_r2:.6f}"
    )
    print(
        f"Água real:            "
        f"{real_water:.4f} mm"
    )
    print(
        f"Água prevista:        "
        f"{predicted_water:.4f} mm"
    )
    print(
        f"Diferença:            "
        f"{water_difference:+.4f} mm"
    )
    print(
        f"Diferença percentual: "
        f"{water_difference_percent:+.2f}%"
    )
    print(
        f"Eventos reais:        "
        f"{y_event.sum():,}"
    )
    print(
        f"Eventos previstos:    "
        f"{predicted_event.sum():,}"
    )

    # -------------------------------------------------------------------------
    # MATRIZ DE CONFUSÃO
    # -------------------------------------------------------------------------

    cm = confusion_matrix(
        y_event,
        predicted_event,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Real_0",
            "Real_1",
        ],
        columns=[
            "Previsto_0",
            "Previsto_1",
        ],
    )

    cm_df.to_csv(
        RESULTS_DIR / "confusion_matrix_test.csv"
    )

    # -------------------------------------------------------------------------
    # PREVISÕES
    # -------------------------------------------------------------------------

    predictions = test[
        [
            "id",
            "timestamp",
            "scenario_id",
            TARGET,
        ]
    ].copy()

    predictions["event_probability"] = (
        probability
    )

    predictions["predicted_event"] = (
        predicted_event
    )

    predictions["predicted_depth_mm"] = (
        predicted_depth
    )

    predictions["depth_error_mm"] = (
        predicted_depth
        - predictions[TARGET]
    )

    predictions.to_csv(
        RESULTS_DIR / "predictions_test_2025.csv",
        index=False,
    )

    return (
        classification_metrics,
        regression_metrics,
        end_to_end_metrics,
    )


# =============================================================================
# SALVAR MODELOS
# =============================================================================

def save_models(
    classifier_models,
    regressor_models,
    best_classifier_name,
    best_regressor_name,
    best_threshold,
):
    print("\n" + "=" * 78)
    print("SALVANDO MODELOS")
    print("=" * 78)

    for name, model in classifier_models.items():
        path = (
            MODELS_DIR
            / f"classifier_{name}.joblib"
        )

        joblib.dump(
            model,
            path,
        )

        print(
            f"Classificador salvo: {path.name}"
        )

    for name, model in regressor_models.items():
        path = (
            MODELS_DIR
            / f"regressor_{name}.joblib"
        )

        joblib.dump(
            model,
            path,
        )

        print(
            f"Regressor salvo: {path.name}"
        )

    # -------------------------------------------------------------------------
    # Configuração do melhor modelo
    # -------------------------------------------------------------------------

    best_classifier_path = (
        MODELS_DIR
        / f"classifier_{best_classifier_name}.joblib"
    )

    best_regressor_path = (
        MODELS_DIR
        / f"regressor_{best_regressor_name}.joblib"
    )

    config = {
        "random_seed": RANDOM_SEED,

        "dataset": str(DATA_PATH),

        "train_period": "2020-01-01 to 2023-12-31",
        "validation_period": "2024-01-01 to 2024-12-31",
        "test_period": "2025-01-01 to 2025-12-31",

        "architecture": "two_stage",

        "classifier": {
            "name": best_classifier_name,
            "threshold": float(best_threshold),
            "path": str(
                best_classifier_path
            ),
        },

        "regressor": {
            "name": best_regressor_name,
            "path": str(
                best_regressor_path
            ),
        },

        "features_numeric": NUMERIC_FEATURES,
        "features_categorical": CATEGORICAL_FEATURES,

        "excluded_columns": [
            TARGET,
            "storage_before_irrigation_mm",
            "storage_after_irrigation_mm",
            "dynamic_lower_bound_percent",
            "scenario_id",
            "storage_capacity_mm",
        ],

        "notes": [
            "Classifier predicts irrigation event yes/no.",
            "Regressor is trained only on positive irrigation events.",
            "Threshold is selected using validation data only.",
            "2025 is reserved as final untouched test.",
            "Target is simulated irrigation depth in mm.",
        ],
    }

    with open(
        MODELS_DIR / "model_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"\nConfiguração salva: "
        f"{MODELS_DIR / 'model_config.json'}"
    )


# =============================================================================
# RELATÓRIO FINAL
# =============================================================================

def save_final_report(
    classification_metrics,
    regression_metrics,
    end_to_end_metrics,
):
    report = {
        "classification_test": classification_metrics,
        "regression_test_events_only": regression_metrics,
        "end_to_end_test": end_to_end_metrics,
    }

    with open(
        RESULTS_DIR / "final_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"Relatório final salvo: "
        f"{RESULTS_DIR / 'final_metrics.json'}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    np.random.seed(RANDOM_SEED)

    # -------------------------------------------------------------------------
    # 1. Dataset
    # -------------------------------------------------------------------------

    df = load_dataset()

    # -------------------------------------------------------------------------
    # 2. Divisão temporal
    # -------------------------------------------------------------------------

    train, validation, test = temporal_split(
        df
    )

    # -------------------------------------------------------------------------
    # 3. Classificação
    # -------------------------------------------------------------------------

    (
        classification_results,
        classifier_models,
        best_classifier_name,
        best_threshold,
    ) = train_classifiers(
        train,
        validation,
    )

    # -------------------------------------------------------------------------
    # 4. Regressão
    # -------------------------------------------------------------------------

    (
        regression_results,
        regressor_models,
        best_regressor_name,
    ) = train_regressors(
        train,
        validation,
    )

    # -------------------------------------------------------------------------
    # 5. Modelos selecionados
    # -------------------------------------------------------------------------

    best_classifier = classifier_models[
        best_classifier_name
    ]

    best_regressor = regressor_models[
        best_regressor_name
    ]

    # -------------------------------------------------------------------------
    # 6. Teste final
    # -------------------------------------------------------------------------

    (
        classification_metrics,
        regression_metrics,
        end_to_end_metrics,
    ) = evaluate_final_model(
        classifier=best_classifier,
        classifier_name=best_classifier_name,
        threshold=best_threshold,
        regressor=best_regressor,
        regressor_name=best_regressor_name,
        test=test,
    )

    # -------------------------------------------------------------------------
    # 7. Salvar modelos
    # -------------------------------------------------------------------------

    save_models(
        classifier_models=classifier_models,
        regressor_models=regressor_models,
        best_classifier_name=best_classifier_name,
        best_regressor_name=best_regressor_name,
        best_threshold=best_threshold,
    )

    # -------------------------------------------------------------------------
    # 8. Salvar relatório final
    # -------------------------------------------------------------------------

    save_final_report(
        classification_metrics,
        regression_metrics,
        end_to_end_metrics,
    )

    # -------------------------------------------------------------------------
    # 9. Resumo
    # -------------------------------------------------------------------------

    print("\n" + "=" * 78)
    print("TREINAMENTO CONCLUÍDO")
    print("=" * 78)

    print(
        f"\nMelhor classificador: "
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

    print(
        f"\nModelos: "
        f"{MODELS_DIR}"
    )

    print(
        f"Resultados: "
        f"{RESULTS_DIR}"
    )

    print("\nArquivos principais:")

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
        "  - final_metrics.json"
    )

    print(
        "  - model_config.json"
    )

    print("\n" + "=" * 78)
    print("FIM")
    print("=" * 78)


if __name__ == "__main__":
    main()