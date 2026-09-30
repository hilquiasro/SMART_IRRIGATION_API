from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "irrigation_dataset_3.csv"

dataset = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------
# Conversões auxiliares
# ---------------------------------------------------------

dataset["et0_daily_mm"] = dataset["et0"] * 24
dataset["etc_daily_mm"] = dataset["etc"] * 24

dataset["raw_depletion_mm"] = (
    dataset["taw_mm"]
    - dataset["minimum_storage_mm"]
)

dataset["below_minimum_storage"] = (
    dataset["storage_after_rain_mm"]
    < dataset["minimum_storage_mm"]
)

dataset["irrigation_event"] = (
    dataset["irrigation_depth_mm"] > 0
)

# ---------------------------------------------------------
# Identificação dos ciclos
# ---------------------------------------------------------

dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])

first_timestamp = dataset["timestamp"].min()

dataset["cycle_id"] = (
    (dataset["timestamp"] - first_timestamp)
    .dt.days // 60
) + 1

# ---------------------------------------------------------
# 1. Evapotranspiração
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("EVAPOTRANSPIRAÇÃO")
print("=" * 70)

print(
    f"ET0 média:       "
    f"{dataset['et0_daily_mm'].mean():.3f} mm/dia"
)

print(
    f"ET0 mediana:     "
    f"{dataset['et0_daily_mm'].median():.3f} mm/dia"
)

print(
    f"ETc média:       "
    f"{dataset['etc_daily_mm'].mean():.3f} mm/dia"
)

print(
    f"ETc mediana:     "
    f"{dataset['etc_daily_mm'].median():.3f} mm/dia"
)

print(
    f"ETc horária:     "
    f"{dataset['etc'].mean():.4f} mm/h"
)

# ---------------------------------------------------------
# 2. Armazenamento
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ARMAZENAMENTO")
print("=" * 70)

print(
    f"TAW média:               "
    f"{dataset['taw_mm'].mean():.3f} mm"
)

print(
    f"Armazenamento médio:     "
    f"{dataset['storage_before_mm'].mean():.3f} mm"
)

print(
    f"Armazenamento após ETc:  "
    f"{dataset['storage_after_et_mm'].mean():.3f} mm"
)

print(
    f"Armazenamento após chuva:"
    f" {dataset['storage_after_rain_mm'].mean():.3f} mm"
)

print(
    f"Armazenamento mínimo:     "
    f"{dataset['minimum_storage_mm'].mean():.3f} mm"
)

print(
    f"RAW médio:                "
    f"{dataset['raw_depletion_mm'].mean():.3f} mm"
)

# ---------------------------------------------------------
# 3. Irrigação
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("IRRIGAÇÃO")
print("=" * 70)

events = dataset.loc[
    dataset["irrigation_event"],
    "irrigation_depth_mm",
]

print(
    f"Registros:              "
    f"{len(dataset):,}"
)

print(
    f"Eventos:                "
    f"{len(events):,}"
)

print(
    f"Taxa de eventos:        "
    f"{len(events) / len(dataset) * 100:.3f}%"
)

print(
    f"Volume total equivalente:"
    f" {events.sum():.2f} mm"
)

if len(events) > 0:
    print(
        f"Evento médio:           "
        f"{events.mean():.4f} mm"
    )

    print(
        f"Mediana:                "
        f"{events.median():.4f} mm"
    )

    print(
        f"P90:                    "
        f"{events.quantile(.90):.4f} mm"
    )

    print(
        f"P95:                    "
        f"{events.quantile(.95):.4f} mm"
    )

    print(
        f"P99:                    "
        f"{events.quantile(.99):.4f} mm"
    )

    print(
        f"Máximo:                 "
        f"{events.max():.4f} mm"
    )

# ---------------------------------------------------------
# 4. Por estágio
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("IRRIGAÇÃO POR ESTÁGIO")
print("=" * 70)

stage_analysis = (
    dataset
    .groupby("crop_stage")
    .agg(
        registros=("irrigation_depth_mm", "size"),
        eventos=("irrigation_event", "sum"),
        taxa_eventos=("irrigation_event", "mean"),
        irrigacao_total_mm=("irrigation_depth_mm", "sum"),
        irrigacao_media_mm=("irrigation_depth_mm", "mean"),
        etc_medio_mm_dia=("etc_daily_mm", "mean"),
        taw_medio_mm=("taw_mm", "mean"),
        raw_medio_mm=("raw_depletion_mm", "mean"),
    )
)

stage_analysis["taxa_eventos"] *= 100

print(stage_analysis.round(4))

# ---------------------------------------------------------
# 5. Por cenário
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("IRRIGAÇÃO POR CENÁRIO")
print("=" * 70)

scenario_analysis = (
    dataset
    .groupby("scenario_id")
    .agg(
        registros=("irrigation_depth_mm", "size"),
        eventos=("irrigation_event", "sum"),
        irrigacao_total_mm=("irrigation_depth_mm", "sum"),
        irrigacao_media_mm=("irrigation_depth_mm", "mean"),
        etc_medio_mm_dia=("etc_daily_mm", "mean"),
        taw_medio_mm=("taw_mm", "mean"),
        raw_medio_mm=("raw_depletion_mm", "mean"),
    )
)

scenario_analysis["taxa_eventos"] = (
    scenario_analysis["eventos"]
    / scenario_analysis["registros"]
    * 100
)

print(scenario_analysis.round(4))

# ---------------------------------------------------------
# 6. Por ciclo
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("IRRIGAÇÃO POR CICLO")
print("=" * 70)

cycle_analysis = (
    dataset
    .groupby(["scenario_id", "cycle_id"])
    .agg(
        registros=("irrigation_depth_mm", "size"),
        eventos=("irrigation_event", "sum"),
        irrigacao_total_mm=("irrigation_depth_mm", "sum"),
        etc_total_mm=("etc", "sum"),
        etc_medio_h=("etc", "mean"),
        taw_medio_mm=("taw_mm", "mean"),
    )
)

cycle_analysis["taxa_eventos"] = (
    cycle_analysis["eventos"]
    / cycle_analysis["registros"]
    * 100
)

cycle_analysis["irrigacao_por_etc"] = (
    cycle_analysis["irrigacao_total_mm"]
    / cycle_analysis["etc_total_mm"]
)

print(cycle_analysis.describe().round(4))

# ---------------------------------------------------------
# 7. Água aplicada x demanda
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RELAÇÃO IRRIGAÇÃO / ETc")
print("=" * 70)

total_irrigation = dataset["irrigation_depth_mm"].sum()
total_etc = dataset["etc"].sum()

print(
    f"Irrigação total: {total_irrigation:.2f} mm"
)

print(
    f"ETc total:       {total_etc:.2f} mm"
)

print(
    f"Irrigação / ETc: "
    f"{total_irrigation / total_etc:.4f}"
)

# ---------------------------------------------------------
# 8. Armazenamento abaixo do mínimo
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ABAIXO DO ARMAZENAMENTO MÍNIMO")
print("=" * 70)

print(
    f"Geral: "
    f"{dataset['below_minimum_storage'].mean() * 100:.3f}%"
)

by_stage = (
    dataset
    .groupby("crop_stage")["below_minimum_storage"]
    .mean()
    * 100
)

print("\nPor estágio:")
print(by_stage.round(3))

# ---------------------------------------------------------
# 9. Eventos muito pequenos
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("EVENTOS MUITO PEQUENOS")
print("=" * 70)

for threshold in [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]:
    count = (
        events < threshold
    ).sum()

    percentage = (
        count / len(events) * 100
        if len(events) > 0
        else 0
    )

    print(
        f"< {threshold:.2f} mm: "
        f"{count:,} eventos "
        f"({percentage:.2f}%)"
    )

# ---------------------------------------------------------
# 10. Verificação do antigo piso de 4 mm
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("VERIFICAÇÃO DO ANTIGO PISO DE 4 MM")
print("=" * 70)

print(
    "Eventos abaixo de 4 mm:",
    (events < 4).sum(),
)

print(
    "Eventos exatamente em 4 mm:",
    (events == 4).sum(),
)

print("\nAuditoria concluída.")