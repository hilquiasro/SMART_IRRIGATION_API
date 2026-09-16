from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = (
    BASE_DIR
    / "results_forecast_current"
)

PLOTS_DIR = (
    BASE_DIR
    / "plots"
    / "forecast_current"
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ARQUIVOS
# ============================================================

METRICS_PATH = (
    RESULTS_DIR
    / "final_metrics.json"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "predictions_test_2025.csv"
)

CONFUSION_PATH = (
    RESULTS_DIR
    / "confusion_matrix_test.csv"
)

COEFFICIENTS_PATH = (
    RESULTS_DIR
    / "regression_coefficients.csv"
)

FEATURE_IMPORTANCE_PATH = (
    RESULTS_DIR
    / "feature_importance.csv"
)


# ============================================================
# CARREGAMENTO
# ============================================================

print("=" * 70)
print("GERAÇÃO DOS GRÁFICOS — FORECAST-CURRENT")
print("=" * 70)


with open(
    METRICS_PATH,
    "r",
    encoding="utf-8",
) as file:

    metrics = json.load(file)


predictions = pd.read_csv(
    PREDICTIONS_PATH,
    parse_dates=["timestamp"],
)


confusion = pd.read_csv(
    CONFUSION_PATH,
    index_col=0,
)


# ============================================================
# 1. MATRIZ DE CONFUSÃO
# ============================================================

print(
    "\n[1/5] Matriz de confusão..."
)


plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    confusion.values,
)

plt.title(
    "Matriz de confusão — Forecast-Current"
)

plt.xlabel(
    "Classe prevista"
)

plt.ylabel(
    "Classe real"
)

plt.xticks(
    [0, 1],
    ["Não irrigar", "Irrigar"],
)

plt.yticks(
    [0, 1],
    ["Não irrigar", "Irrigar"],
)


for i in range(
    confusion.shape[0]
):

    for j in range(
        confusion.shape[1]
    ):

        plt.text(
            j,
            i,
            f"{confusion.iloc[i, j]:,}",
            ha="center",
            va="center",
        )


plt.tight_layout()

plt.savefig(
    PLOTS_DIR
    / "01_matriz_confusao.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 2. REAL VS PREVISTO — LÂMINA
# ============================================================

print(
    "[2/5] Real vs previsto..."
)


real_events = (
    predictions[
        "irrigation_event_current"
    ] == 1
)

predicted_events = (
    predictions[
        "predicted_event"
    ] == 1
)


# Para avaliar a capacidade de estimar
# a lâmina quando ambos os modelos
# concordam que deve haver irrigação.

mask = (
    real_events
    & predicted_events
)


real_depth = predictions.loc[
    mask,
    "irrigation_depth_current"
]

predicted_depth = predictions.loc[
    mask,
    "predicted_depth_mm"
]


plt.figure(
    figsize=(8, 7)
)

plt.scatter(
    real_depth,
    predicted_depth,
    alpha=0.25,
    s=12,
)


if len(real_depth) > 0:

    min_value = min(
        real_depth.min(),
        predicted_depth.min(),
    )

    max_value = max(
        real_depth.max(),
        predicted_depth.max(),
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        linewidth=2,
        label="Ideal: previsto = real",
    )

    plt.legend()


plt.title(
    "Lâmina real vs prevista — Forecast-Current"
)

plt.xlabel(
    "Lâmina real atual (mm)"
)

plt.ylabel(
    "Lâmina prevista (mm)"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR
    / "02_lamina_real_vs_prevista.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 3. DISTRIBUIÇÃO DA LÂMINA
# ============================================================

print(
    "[3/5] Distribuição das lâminas..."
)


plt.figure(
    figsize=(9, 6)
)


plt.hist(
    predictions[
        "irrigation_depth_current"
    ],
    bins=40,
    alpha=0.65,
    label="Real",
)


plt.hist(
    predictions[
        "predicted_depth_mm"
    ],
    bins=40,
    alpha=0.65,
    label="Prevista",
)


plt.title(
    "Distribuição das lâminas de irrigação"
)

plt.xlabel(
    "Lâmina de irrigação (mm)"
)

plt.ylabel(
    "Frequência"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR
    / "03_distribuicao_lamina.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 4. VARIÁVEIS MAIS RELACIONADAS À REGRESSÃO
# ============================================================

print(
    "[4/5] Variáveis do modelo..."
)


if COEFFICIENTS_PATH.exists():

    coefficients = pd.read_csv(
        COEFFICIENTS_PATH
    )

    coefficients = (
        coefficients
        .sort_values(
            "absolute_coefficient",
            ascending=False,
        )
        .head(20)
        .sort_values(
            "coefficient"
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        coefficients["feature"],
        coefficients["coefficient"],
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.title(
        "Coeficientes da regressão linear — Forecast-Current"
    )

    plt.xlabel(
        "Coeficiente padronizado"
    )

    plt.ylabel(
        "Variável"
    )

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "04_coeficientes_regressao.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


    print(
        "  Usando regression_coefficients.csv"
    )


elif FEATURE_IMPORTANCE_PATH.exists():

    # Fallback para o caso de futuramente
    # o melhor regressor ser um modelo de árvore.

    importance = pd.read_csv(
        FEATURE_IMPORTANCE_PATH
    )

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .head(20)
        .sort_values(
            "importance"
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        importance["feature"],
        importance["importance"],
    )

    plt.title(
        "Importância das variáveis — Forecast-Current"
    )

    plt.xlabel(
        "Importância"
    )

    plt.ylabel(
        "Variável"
    )

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "04_importancia_variaveis.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


    print(
        "  Usando feature_importance.csv"
    )


else:

    print(
        "  AVISO: nenhum arquivo de análise de variáveis encontrado."
    )


# ============================================================
# 5. ERRO DA REGRESSÃO
# ============================================================

print(
    "[5/5] Erro da regressão..."
)


regression_mask = (
    predictions[
        "irrigation_event_current"
    ] == 1
)


real_regression = predictions.loc[
    regression_mask,
    "irrigation_depth_current"
]

predicted_regression = predictions.loc[
    regression_mask,
    "predicted_depth_mm"
]


errors = (
    predicted_regression
    - real_regression
)


plt.figure(
    figsize=(9, 6)
)

plt.hist(
    errors,
    bins=40,
    alpha=0.75,
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=2,
)


plt.title(
    "Erro da estimativa de lâmina — Forecast-Current"
)

plt.xlabel(
    "Erro: lâmina prevista − lâmina real (mm)"
)

plt.ylabel(
    "Frequência"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR
    / "05_erro_regressao.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# RESUMO
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "GRÁFICOS GERADOS"
)

print(
    "=" * 70
)

for plot in sorted(
    PLOTS_DIR.glob("*.png")
):

    print(
        f"  ✓ {plot.name}"
    )


print(
    f"\nPasta:"
)

print(
    PLOTS_DIR
)

print(
    "\n" + "=" * 70
)