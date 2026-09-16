from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results_8"
PREDICTIONS_FILE = RESULTS_DIR / "predictions_test_2025.csv"
OUTPUT_DIR = RESULTS_DIR / "error_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def find_column(df, candidates):
    """
    Encontra a primeira coluna existente dentre as candidatas.
    """
    for column in candidates:
        if column in df.columns:
            return column
    return None


def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# CARREGAMENTO
# ============================================================

print("=" * 78)
print("ANÁLISE DE ERROS — DATASET 3 — RODADA 8")
print("=" * 78)

print(f"Arquivo: {PREDICTIONS_FILE}")

if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado:\n{PREDICTIONS_FILE}"
    )

df = pd.read_csv(PREDICTIONS_FILE)

print(f"Registros carregados: {len(df):,}")
print(f"Colunas: {len(df.columns)}")

print("\nColunas disponíveis:")
for column in df.columns:
    print(f"  - {column}")


# ============================================================
# IDENTIFICAÇÃO DAS COLUNAS
# ============================================================

timestamp_col = find_column(
    df,
    ["timestamp", "date", "datetime"]
)

stage_col = find_column(
    df,
    ["crop_stage"]
)

moisture_col = find_column(
    df,
    ["soil_moisture_percent"]
)

real_target_col = find_column(
    df,
    [
        "irrigation_depth_mm",
        "y_true",
        "target",
        "real_irrigation_depth_mm",
    ]
)

pred_target_col = find_column(
    df,
    [
        "predicted_irrigation_depth_mm",
        "y_pred",
        "prediction",
        "predicted_depth_mm",
    ]
)

real_class_col = find_column(
    df,
    [
        "irrigation_event",
        "real_event",
        "y_true_class",
        "target_class",
    ]
)

pred_class_col = find_column(
    df,
    [
        "predicted_event",
        "prediction_class",
        "y_pred_class",
    ]
)

probability_col = find_column(
    df,
    [
        "irrigation_probability",
        "predicted_probability",
        "probability",
        "y_prob",
    ]
)

hour_col = find_column(
    df,
    ["hour"]
)


# ============================================================
# TENTATIVA DE DERIVAR COLUNAS AUSENTES
# ============================================================

if timestamp_col is not None:

    df[timestamp_col] = pd.to_datetime(
        df[timestamp_col],
        errors="coerce"
    )

    if hour_col is None:
        df["analysis_hour"] = df[timestamp_col].dt.hour
        hour_col = "analysis_hour"


# Se as classes não estiverem explícitas, derivamos do target.

if real_class_col is None and real_target_col is not None:
    df["analysis_real_event"] = (
        df[real_target_col] > 0
    ).astype(int)

    real_class_col = "analysis_real_event"


if pred_class_col is None and pred_target_col is not None:
    df["analysis_predicted_event"] = (
        df[pred_target_col] > 0
    ).astype(int)

    pred_class_col = "analysis_predicted_event"


# ============================================================
# VERIFICAÇÃO
# ============================================================

required = {
    "target real": real_target_col,
    "previsão": pred_target_col,
    "classe real": real_class_col,
    "classe prevista": pred_class_col,
}

missing = [
    name
    for name, column in required.items()
    if column is None
]

if missing:
    raise ValueError(
        "\nNão foi possível identificar as seguintes informações:\n"
        + "\n".join(f"  - {item}" for item in missing)
        + "\n\nVerifique os nomes das colunas do arquivo."
    )


# ============================================================
# NORMALIZAÇÃO
# ============================================================

df["real_depth_mm"] = pd.to_numeric(
    df[real_target_col],
    errors="coerce"
)

df["predicted_depth_mm"] = pd.to_numeric(
    df[pred_target_col],
    errors="coerce"
)

df["real_event"] = pd.to_numeric(
    df[real_class_col],
    errors="coerce"
).fillna(0).astype(int)

df["predicted_event"] = pd.to_numeric(
    df[pred_class_col],
    errors="coerce"
).fillna(0).astype(int)

df["depth_error_mm"] = (
    df["predicted_depth_mm"] -
    df["real_depth_mm"]
)

df["absolute_depth_error_mm"] = (
    df["depth_error_mm"].abs()
)

df["squared_depth_error_mm"] = (
    df["depth_error_mm"] ** 2
)

df["error_type"] = "true_negative"

df.loc[
    (df["real_event"] == 0) &
    (df["predicted_event"] == 1),
    "error_type"
] = "false_positive"

df.loc[
    (df["real_event"] == 1) &
    (df["predicted_event"] == 0),
    "error_type"
] = "false_negative"

df.loc[
    (df["real_event"] == 1) &
    (df["predicted_event"] == 1),
    "error_type"
] = "true_positive"


# ============================================================
# 1. MATRIZ DE CONFUSÃO
# ============================================================

print_section("1. MATRIZ DE CONFUSÃO")

cm = confusion_matrix(
    df["real_event"],
    df["predicted_event"],
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print(f"True Negative : {tn:,}")
print(f"False Positive: {fp:,}")
print(f"False Negative: {fn:,}")
print(f"True Positive : {tp:,}")

cm_df = pd.DataFrame(
    cm,
    index=["Real: não irrigar", "Real: irrigar"],
    columns=["Previsto: não irrigar", "Previsto: irrigar"],
)

cm_df.to_csv(
    OUTPUT_DIR / "confusion_matrix.csv"
)


# ============================================================
# 2. MÉTRICAS
# ============================================================

print_section("2. MÉTRICAS DE CLASSIFICAÇÃO")

report = classification_report(
    df["real_event"],
    df["predicted_event"],
    labels=[0, 1],
    target_names=["Não irrigar", "Irrigar"],
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()

print(
    report_df[
        ["precision", "recall", "f1-score", "support"]
    ].round(4)
)

report_df.to_csv(
    OUTPUT_DIR / "classification_report.csv"
)


# ============================================================
# 3. ANÁLISE DE EVENTOS
# ============================================================

print_section("3. ANÁLISE DOS EVENTOS")

real_events = df[df["real_event"] == 1]
predicted_events = df[df["predicted_event"] == 1]
false_positives = df[df["error_type"] == "false_positive"]
false_negatives = df[df["error_type"] == "false_negative"]
true_positives = df[df["error_type"] == "true_positive"]

print(f"Eventos reais:       {len(real_events):,}")
print(f"Eventos previstos:   {len(predicted_events):,}")
print(f"True positives:      {len(true_positives):,}")
print(f"False positives:     {len(false_positives):,}")
print(f"False negatives:     {len(false_negatives):,}")


# ============================================================
# 4. REGRESSÃO
# ============================================================

print_section("4. ERRO DA ESTIMATIVA DA LÂMINA")

event_regression = df[
    (df["real_event"] == 1) &
    (df["predicted_event"] == 1)
].copy()

if len(event_regression) > 0:

    mae = mean_absolute_error(
        event_regression["real_depth_mm"],
        event_regression["predicted_depth_mm"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            event_regression["real_depth_mm"],
            event_regression["predicted_depth_mm"]
        )
    )

    r2 = r2_score(
        event_regression["real_depth_mm"],
        event_regression["predicted_depth_mm"]
    )

    print(f"Eventos TP analisados: {len(event_regression):,}")
    print(f"MAE : {mae:.6f} mm")
    print(f"RMSE: {rmse:.6f} mm")
    print(f"R²  : {r2:.6f}")

    event_regression[
        [
            "real_depth_mm",
            "predicted_depth_mm",
            "depth_error_mm",
            "absolute_depth_error_mm",
        ]
    ].describe().to_csv(
        OUTPUT_DIR / "event_regression_statistics.csv"
    )


# ============================================================
# 5. ÁGUA ACUMULADA
# ============================================================

print_section("5. ÁGUA ACUMULADA")

real_water = df["real_depth_mm"].sum()
predicted_water = (
    df["predicted_depth_mm"]
    .where(df["predicted_event"] == 1, 0)
    .sum()
)

water_difference = predicted_water - real_water

water_difference_percent = (
    water_difference / real_water * 100
    if real_water != 0
    else np.nan
)

print(f"Água real:       {real_water:.4f} mm")
print(f"Água prevista:   {predicted_water:.4f} mm")
print(f"Diferença:       {water_difference:+.4f} mm")
print(f"Diferença (%):   {water_difference_percent:+.4f}%")


# ============================================================
# 6. ERROS POR TIPO
# ============================================================

print_section("6. CARACTERIZAÇÃO DOS ERROS")

error_summary = (
    df.groupby("error_type")
    .agg(
        records=("error_type", "size"),
        mean_absolute_depth_error=(
            "absolute_depth_error_mm",
            "mean"
        ),
        mean_depth_error=(
            "depth_error_mm",
            "mean"
        ),
    )
    .reset_index()
)

print(error_summary.round(4))

error_summary.to_csv(
    OUTPUT_DIR / "error_summary.csv",
    index=False
)


# ============================================================
# 7. FALSOS NEGATIVOS
# ============================================================

print_section("7. FALSOS NEGATIVOS")

if len(false_negatives) > 0:

    print(
        false_negatives[
            [
                column
                for column in [
                    timestamp_col,
                    stage_col,
                    moisture_col,
                    real_target_col,
                    pred_target_col,
                    probability_col,
                ]
                if column is not None
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    false_negatives.to_csv(
        OUTPUT_DIR / "false_negatives.csv",
        index=False
    )

else:
    print("Nenhum falso negativo.")


# ============================================================
# 8. FALSOS POSITIVOS
# ============================================================

print_section("8. FALSOS POSITIVOS")

if len(false_positives) > 0:

    print(
        false_positives[
            [
                column
                for column in [
                    timestamp_col,
                    stage_col,
                    moisture_col,
                    real_target_col,
                    pred_target_col,
                    probability_col,
                ]
                if column is not None
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    false_positives.to_csv(
        OUTPUT_DIR / "false_positives.csv",
        index=False
    )

else:
    print("Nenhum falso positivo.")


# ============================================================
# 9. ERROS POR ESTÁGIO
# ============================================================

if stage_col is not None:

    print_section("9. ERROS POR ESTÁGIO DO COENTRO")

    stage_analysis = (
        df.groupby(stage_col)
        .agg(
            registros=("real_event", "size"),
            eventos_reais=("real_event", "sum"),
            eventos_previstos=("predicted_event", "sum"),
            falsos_positivos=(
                "error_type",
                lambda x: (x == "false_positive").sum()
            ),
            falsos_negativos=(
                "error_type",
                lambda x: (x == "false_negative").sum()
            ),
            verdadeiros_positivos=(
                "error_type",
                lambda x: (x == "true_positive").sum()
            ),
        )
        .reset_index()
    )

    stage_analysis["event_rate_real_percent"] = (
        stage_analysis["eventos_reais"] /
        stage_analysis["registros"] *
        100
    )

    stage_analysis["event_rate_predicted_percent"] = (
        stage_analysis["eventos_previstos"] /
        stage_analysis["registros"] *
        100
    )

    print(
        stage_analysis.round(4).to_string(index=False)
    )

    stage_analysis.to_csv(
        OUTPUT_DIR / "errors_by_crop_stage.csv",
        index=False
    )


# ============================================================
# 10. ERROS POR FAIXA DE UMIDADE
# ============================================================

if moisture_col is not None:

    print_section("10. ERROS POR FAIXA DE UMIDADE DO SOLO")

    df["moisture_bin"] = pd.cut(
        pd.to_numeric(
            df[moisture_col],
            errors="coerce"
        ),
        bins=[
            0,
            45,
            48,
            50,
            52,
            54,
            56,
            60,
            100,
        ],
        include_lowest=True
    )

    moisture_analysis = (
        df.groupby(
            "moisture_bin",
            observed=True
        )
        .agg(
            registros=("real_event", "size"),
            eventos_reais=("real_event", "sum"),
            eventos_previstos=("predicted_event", "sum"),
            falsos_positivos=(
                "error_type",
                lambda x: (x == "false_positive").sum()
            ),
            falsos_negativos=(
                "error_type",
                lambda x: (x == "false_negative").sum()
            ),
        )
        .reset_index()
    )

    print(
        moisture_analysis.to_string(index=False)
    )

    moisture_analysis.to_csv(
        OUTPUT_DIR / "errors_by_soil_moisture.csv",
        index=False
    )


# ============================================================
# 11. ERROS POR HORÁRIO
# ============================================================

if hour_col is not None:

    print_section("11. ERROS POR HORÁRIO")

    hour_analysis = (
        df.groupby(hour_col)
        .agg(
            registros=("real_event", "size"),
            eventos_reais=("real_event", "sum"),
            eventos_previstos=("predicted_event", "sum"),
            falsos_positivos=(
                "error_type",
                lambda x: (x == "false_positive").sum()
            ),
            falsos_negativos=(
                "error_type",
                lambda x: (x == "false_negative").sum()
            ),
        )
        .reset_index()
        .sort_values(hour_col)
    )

    print(
        hour_analysis.to_string(index=False)
    )

    hour_analysis.to_csv(
        OUTPUT_DIR / "errors_by_hour.csv",
        index=False
    )


# ============================================================
# 12. MAIORES ERROS DE LÂMINA
# ============================================================

print_section("12. MAIORES ERROS DE LÂMINA")

largest_errors = (
    df[
        (df["real_event"] == 1) |
        (df["predicted_event"] == 1)
    ]
    .sort_values(
        "absolute_depth_error_mm",
        ascending=False
    )
)

columns_to_show = [
    column
    for column in [
        timestamp_col,
        stage_col,
        moisture_col,
        real_target_col,
        pred_target_col,
        probability_col,
        "error_type",
        "depth_error_mm",
        "absolute_depth_error_mm",
    ]
    if column is not None or column in df.columns
]

columns_to_show = [
    column
    for column in columns_to_show
    if column in df.columns
]

print(
    largest_errors[
        columns_to_show
    ]
    .head(20)
    .to_string(index=False)
)

largest_errors.to_csv(
    OUTPUT_DIR / "largest_depth_errors.csv",
    index=False
)


# ============================================================
# 13. ERRO POR EVENTO REAL
# ============================================================

print_section("13. ERRO POR EVENTO REAL")

real_event_analysis = df[
    df["real_event"] == 1
].copy()

real_event_analysis["detected"] = (
    real_event_analysis["predicted_event"] == 1
)

real_event_analysis["water_difference_mm"] = (
    real_event_analysis["predicted_depth_mm"] -
    real_event_analysis["real_depth_mm"]
)

real_event_columns = [
    column
    for column in [
        timestamp_col,
        stage_col,
        moisture_col,
        "real_depth_mm",
        "predicted_depth_mm",
        "water_difference_mm",
        "absolute_depth_error_mm",
        "detected",
    ]
    if column is not None and column in real_event_analysis.columns
]

real_event_analysis[
    real_event_columns
].to_csv(
    OUTPUT_DIR / "real_event_analysis.csv",
    index=False
)


# ============================================================
# 14. RESUMO JSON
# ============================================================

summary = {
    "records": int(len(df)),
    "real_events": int(df["real_event"].sum()),
    "predicted_events": int(df["predicted_event"].sum()),
    "true_positives": int(tp),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_negatives": int(tn),
    "real_water_mm": float(real_water),
    "predicted_water_mm": float(predicted_water),
    "water_difference_mm": float(water_difference),
    "water_difference_percent": float(
        water_difference_percent
    ),
}

if len(event_regression) > 0:
    summary.update(
        {
            "event_regression_mae_mm": float(mae),
            "event_regression_rmse_mm": float(rmse),
            "event_regression_r2": float(r2),
        }
    )


with open(
    OUTPUT_DIR / "error_analysis_summary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# FINAL
# ============================================================

print_section("ANÁLISE CONCLUÍDA")

print(f"Resultados salvos em:")
print(f"{OUTPUT_DIR}")

print("\nPrincipais arquivos:")
print("  - confusion_matrix.csv")
print("  - classification_report.csv")
print("  - error_summary.csv")
print("  - false_positives.csv")
print("  - false_negatives.csv")
print("  - errors_by_crop_stage.csv")
print("  - errors_by_soil_moisture.csv")
print("  - errors_by_hour.csv")
print("  - largest_depth_errors.csv")
print("  - real_event_analysis.csv")
print("  - error_analysis_summary.json")