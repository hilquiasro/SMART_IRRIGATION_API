from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots_forecast"

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CARREGAMENTO DOS RESULTADOS
# ============================================================

predictions_path = (
    RESULTS_DIR
    / "forecast_predictions_test_2025.csv"
)

confusion_path = (
    RESULTS_DIR
    / "forecast_confusion_matrix_test.csv"
)

feature_importance_path = (
    RESULTS_DIR
    / "forecast_feature_importance.csv"
)

metrics_path = (
    RESULTS_DIR
    / "forecast_final_metrics.json"
)

classification_results_path = (
    RESULTS_DIR
    / "forecast_classification_results.csv"
)

regression_results_path = (
    RESULTS_DIR
    / "forecast_regression_results.csv"
)


predictions = pd.read_csv(
    predictions_path
)

confusion = pd.read_csv(
    confusion_path,
    index_col=0
)

feature_importance = pd.read_csv(
    feature_importance_path
)

classification_results = pd.read_csv(
    classification_results_path
)

regression_results = pd.read_csv(
    regression_results_path
)


with open(
    metrics_path,
    "r",
    encoding="utf-8"
) as f:

    metrics = json.load(f)


# ============================================================
# ESTILO
# ============================================================

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


# ============================================================
# INFORMAÇÕES DO MODELO
# ============================================================

classifier_name = metrics[
    "classifier"
]

regressor_name = metrics[
    "regressor"
]

threshold = metrics[
    "classification_threshold"
]


# ============================================================
# 1. MATRIZ DE CONFUSÃO
# ============================================================

real_event = (
    predictions[
        "irrigation_event_next_24h"
    ] == 1
)

predicted_event = (
    predictions[
        "predicted_irrigation_event"
    ] == 1
)

tn = int(
    (
        ~real_event
        & ~predicted_event
    ).sum()
)

fp = int(
    (
        ~real_event
        & predicted_event
    ).sum()
)

fn = int(
    (
        real_event
        & ~predicted_event
    ).sum()
)

tp = int(
    (
        real_event
        & predicted_event
    ).sum()
)

matrix = [
    [tn, fp],
    [fn, tp],
]


fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.imshow(matrix)

ax.set_xticks(
    [0, 1]
)

ax.set_yticks(
    [0, 1]
)

ax.set_xticklabels([
    "Não irrigar",
    "Irrigar"
])

ax.set_yticklabels([
    "Não irrigar",
    "Irrigar"
])

ax.set_xlabel(
    "Classe prevista"
)

ax.set_ylabel(
    "Classe real"
)

ax.set_title(
    "Matriz de confusão — "
    f"{classifier_name} no teste de 2025"
)


for i in range(2):

    for j in range(2):

        ax.text(
            j,
            i,
            f"{matrix[i][j]:,}".replace(
                ",",
                "."
            ),
            ha="center",
            va="center",
            fontsize=16,
        )


plt.tight_layout()

plt.savefig(
    PLOTS_DIR
    / "01_matriz_confusao.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print()
print("Matriz de confusão:")
print(f"TN = {tn}")
print(f"FP = {fp}")
print(f"FN = {fn}")
print(f"TP = {tp}")


# ============================================================
# 2. ÁGUA ACUMULADA — REFERÊNCIA × MODELO
# ============================================================

predictions["timestamp"] = pd.to_datetime(
    predictions["timestamp"]
)

predictions = (
    predictions
    .sort_values("timestamp")
    .copy()
)


daily_water = (
    predictions
    .groupby(
        predictions["timestamp"].dt.date
    )
    .agg(
        real_water=(
            "irrigation_depth_next_24h",
            "sum"
        ),
        predicted_water=(
            "predicted_irrigation_depth_end_to_end_mm",
            "sum"
        ),
    )
    .reset_index()
)


daily_water["timestamp"] = pd.to_datetime(
    daily_water["timestamp"]
)


daily_water["real_cumulative"] = (
    daily_water[
        "real_water"
    ].cumsum()
)

daily_water["predicted_cumulative"] = (
    daily_water[
        "predicted_water"
    ].cumsum()
)


real_water = (
    daily_water[
        "real_cumulative"
    ].iloc[-1]
)

predicted_water = (
    daily_water[
        "predicted_cumulative"
    ].iloc[-1]
)


difference_mm = (
    predicted_water
    - real_water
)

difference_percent = (
    difference_mm
    / real_water
) * 100


fig, ax = plt.subplots(
    figsize=(11, 6)
)


ax.plot(
    daily_water["timestamp"],
    daily_water["real_cumulative"],
    label="Política de referência",
    linewidth=2,
)


ax.plot(
    daily_water["timestamp"],
    daily_water["predicted_cumulative"],
    label="Modelo",
    linewidth=2,
)


ax.scatter(
    daily_water["timestamp"].iloc[-1],
    real_water,
    s=50,
)


ax.scatter(
    daily_water["timestamp"].iloc[-1],
    predicted_water,
    s=50,
)


ax.set_xlabel(
    "Data"
)

ax.set_ylabel(
    "Lâmina acumulada de irrigação (mm)"
)

ax.set_title(
    "Lâmina de irrigação acumulada — "
    "teste de 2025"
)


ax.legend()


ax.text(
    0.02,
    0.96,
    (
        f"Referência: {real_water:.2f} mm\n"
        f"Modelo: {predicted_water:.2f} mm\n"
        f"Diferença: {difference_mm:+.2f} mm "
        f"({difference_percent:+.2f}%)"
    ),
    transform=ax.transAxes,
    verticalalignment="top",
)


plt.tight_layout()

plt.savefig(
    PLOTS_DIR
    / "02_agua_acumulada_real_vs_prevista.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print()
print("Água acumulada:")
print(
    f"Referência: {real_water:.4f} mm"
)
print(
    f"Modelo:     {predicted_water:.4f} mm"
)
print(
    f"Diferença:  {difference_mm:+.4f} mm"
)
print(
    f"Percentual:  {difference_percent:+.2f}%"
)


# ============================================================
# 3. REAL × PREVISTO DA LÂMINA NOS EVENTOS
# ============================================================

tp_data = predictions[
    (
        predictions[
            "predicted_irrigation_event"
        ] == 1
    )
    &
    (
        predictions[
            "irrigation_event_next_24h"
        ] == 1
    )
].copy()


fig, ax = plt.subplots(
    figsize=(8, 6)
)


ax.scatter(
    tp_data[
        "irrigation_depth_next_24h"
    ],
    tp_data[
        "predicted_irrigation_depth_mm"
    ],
    alpha=0.15,
    s=12,
)


min_value = min(
    tp_data[
        "irrigation_depth_next_24h"
    ].min(),

    tp_data[
        "predicted_irrigation_depth_mm"
    ].min(),
)


max_value = max(
    tp_data[
        "irrigation_depth_next_24h"
    ].max(),

    tp_data[
        "predicted_irrigation_depth_mm"
    ].max(),
)


ax.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
)


ax.set_xlabel(
    "Lâmina real nas próximas 24 h (mm)"
)

ax.set_ylabel(
    "Lâmina prevista (mm)"
)

ax.set_title(
    "Lâmina real × prevista "
    "nos eventos detectados"
)


mae_tp = (
    tp_data[
        "absolute_error_mm"
    ].mean()
)


ax.text(
    0.05,
    0.95,
    f"MAE = {mae_tp:.4f} mm",
    transform=ax.transAxes,
    va="top",
)


plt.tight_layout()

plt.savefig(
    PLOTS_DIR
    / "03_real_vs_previsto_lamina.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 4. ERROS POR FAIXA DE UMIDADE DO SOLO
# ============================================================

bins = [
    -float("inf"),
    45,
    48,
    50,
    52,
    54,
    56,
    float("inf")
]

labels = [
    "≤45%",
    "45–48%",
    "48–50%",
    "50–52%",
    "52–54%",
    "54–56%",
    ">56%",
]


predictions["soil_moisture_range"] = pd.cut(
    predictions[
        "soil_moisture_percent"
    ],
    bins=bins,
    labels=labels,
    right=True,
)


grouped = []


for label in labels:

    group = predictions[
        predictions[
            "soil_moisture_range"
        ] == label
    ]


    false_positive = (
        (
            group[
                "irrigation_event_next_24h"
            ] == 0
        )
        &
        (
            group[
                "predicted_irrigation_event"
            ] == 1
        )
    ).sum()


    false_negative = (
        (
            group[
                "irrigation_event_next_24h"
            ] == 1
        )
        &
        (
            group[
                "predicted_irrigation_event"
            ] == 0
        )
    ).sum()


    grouped.append({
        "faixa": label,
        "FP": false_positive,
        "FN": false_negative,
    })


error_df = pd.DataFrame(
    grouped
)


fig, ax = plt.subplots(
    figsize=(10, 6)
)


x = range(
    len(error_df)
)

width = 0.38


ax.bar(
    [
        i - width / 2
        for i in x
    ],
    error_df["FP"],
    width,
    label="Falsos positivos",
)


ax.bar(
    [
        i + width / 2
        for i in x
    ],
    error_df["FN"],
    width,
    label="Falsos negativos",
)


ax.set_xticks(
    list(x)
)

ax.set_xticklabels(
    error_df["faixa"]
)


ax.set_xlabel(
    "Umidade do solo no momento da decisão"
)

ax.set_ylabel(
    "Quantidade de erros"
)

ax.set_title(
    "Erros de classificação por "
    "faixa de umidade do solo"
)


ax.legend()


plt.tight_layout()

plt.savefig(
    PLOTS_DIR
    / "04_erros_por_umidade.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 5. IMPORTÂNCIA DAS FEATURES
# ============================================================

feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=True
    )
)


fig, ax = plt.subplots(
    figsize=(10, 8)
)


ax.barh(
    feature_importance[
        "feature"
    ],
    feature_importance[
        "importance"
    ],
)


ax.set_xlabel(
    "Importância"
)

ax.set_ylabel(
    "Variável"
)

ax.set_title(
    "Importância das variáveis — "
    f"{regressor_name}"
)


plt.tight_layout()

plt.savefig(
    PLOTS_DIR
    / "05_importancia_features.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 6. EVENTOS POR ESTÁGIO DO COENTRO
# ============================================================

stages = [
    "initial",
    "development",
    "mid",
    "late",
]


stage_names = {
    "initial": "Inicial",
    "development": "Desenvolvimento",
    "mid": "Meio",
    "late": "Final",
}


stage_data = []


for stage in stages:

    group = predictions[
        predictions[
            "crop_stage"
        ] == stage
    ]


    real_events = (
        group[
            "irrigation_event_next_24h"
        ] == 1
    ).sum()


    predicted_events = (
        group[
            "predicted_irrigation_event"
        ] == 1
    ).sum()


    stage_data.append({
        "stage": stage_names[stage],
        "real": real_events,
        "predicted": predicted_events,
    })


stage_df = pd.DataFrame(
    stage_data
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)


x = range(
    len(stage_df)
)

width = 0.38


ax.bar(
    [
        i - width / 2
        for i in x
    ],
    stage_df["real"],
    width,
    label="Eventos de referência",
)


ax.bar(
    [
        i + width / 2
        for i in x
    ],
    stage_df["predicted"],
    width,
    label="Eventos previstos",
)


ax.set_xticks(
    list(x)
)

ax.set_xticklabels(
    stage_df["stage"]
)


ax.set_xlabel(
    "Estágio fenológico"
)

ax.set_ylabel(
    "Número de eventos"
)

ax.set_title(
    "Eventos de irrigação por "
    "estágio do coentro"
)


ax.legend()


plt.tight_layout()

plt.savefig(
    PLOTS_DIR
    / "06_eventos_por_estagio.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# RESUMO
# ============================================================

print()
print("=" * 60)
print(
    "GRÁFICOS DO EXPERIMENTO FORECAST "
    "GERADOS COM SUCESSO"
)
print("=" * 60)

print()
print(
    f"Pasta: {PLOTS_DIR}"
)

print()
print("Arquivos:")

for file in sorted(
    PLOTS_DIR.glob("*.png")
):

    print(
        f" - {file.name}"
    )


print()
print(
    f"Classificador: {classifier_name}"
)

print(
    f"Regressor:     {regressor_name}"
)

print(
    f"Threshold:     {threshold:.2f}"
)

print()
print(
    f"Água real:     {real_water:.4f} mm"
)

print(
    f"Água prevista: {predicted_water:.4f} mm"
)

print(
    f"Diferença:     {difference_percent:+.2f}%"
)

print()
print(
    f"Eventos reais:     {real_event.sum():,}"
)

print(
    f"Eventos previstos: {predicted_event.sum():,}"
)

print()
print(
    "Nenhum modelo foi treinado novamente."
)

print(
    "Os gráficos foram gerados somente a partir "
    "dos resultados do experimento Forecast."
)

print()