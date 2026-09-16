from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

DATA_DIR = Path("ml/data")

DATASET_FILE = (
    DATA_DIR / "irrigation_dataset_forecast.csv"
)


# ============================================================
# COLUNAS
# ============================================================

FORECAST_COLS = [

    # +24h
    "forecast_temperature_24h_c",
    "forecast_humidity_24h_percent",
    "forecast_precipitation_24h_mm",
    "forecast_wind_24h_m_s",
    "forecast_radiation_24h_w_m2",
    "forecast_et0_24h_mm",

    # +48h
    "forecast_temperature_48h_c",
    "forecast_humidity_48h_percent",
    "forecast_precipitation_48h_mm",
    "forecast_wind_48h_m_s",
    "forecast_radiation_48h_w_m2",
    "forecast_et0_48h_mm",

    # +72h
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


HISTORICAL_FEATURES = [

    "soil_moisture_percent",

    "soil_moisture_lag_1h",
    "soil_moisture_lag_3h",
    "soil_moisture_lag_6h",

    "rain_6h",
    "rain_24h",

    "etc",
    "etc_6h",
    "etc_24h",

    "temperature_6h_mean",
    "radiation_6h_sum",

    "kc",
    "et0",

    "hour_sin",
    "hour_cos",

    "day_of_year_sin",
    "day_of_year_cos",
]


TARGET_COLS = [

    "irrigation_depth_next_24h",

    "irrigation_event_next_24h",
]


LEAKAGE_COLS = [

    "irrigation_depth_mm",

    "storage_after_irrigation_mm",

    "storage_before_irrigation_mm",

    "dynamic_lower_bound_percent",

    "storage_capacity_mm",

    "id",
    "scenario_id",
]


# ============================================================
# AUXILIARES
# ============================================================

def print_header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{path}"
        )


# ============================================================
# 1. CARREGAMENTO
# ============================================================

print_header(
    "1. CARREGANDO DATASET FORECAST"
)

check_file(DATASET_FILE)

df = pd.read_csv(
    DATASET_FILE
)

print(
    f"Arquivo: {DATASET_FILE}"
)

print(
    f"Registros: {len(df):,}"
)

print(
    f"Colunas: {len(df.columns)}"
)


# ============================================================
# 2. TIMESTAMP
# ============================================================

print_header(
    "2. VERIFICAÇÃO DO TIMESTAMP"
)

if "timestamp" not in df.columns:

    raise ValueError(
        "Coluna 'timestamp' não encontrada."
    )


df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)


invalid_timestamp = (
    df["timestamp"].isna().sum()
)


print(
    f"Timestamps inválidos: "
    f"{invalid_timestamp:,}"
)


if invalid_timestamp:

    print(
        "❌ Existem timestamps inválidos."
    )

else:

    print(
        "✅ Todos os timestamps são válidos."
    )


print(
    f"Início: {df['timestamp'].min()}"
)

print(
    f"Fim: {df['timestamp'].max()}"
)


# ============================================================
# 3. DUPLICATAS
# ============================================================

print_header(
    "3. DUPLICATAS"
)


duplicate_rows = (
    df.duplicated().sum()
)


duplicate_keys = (
    df.duplicated(
        subset=[
            "timestamp",
            "scenario_id"
        ]
    ).sum()
)


print(
    f"Linhas duplicadas: "
    f"{duplicate_rows:,}"
)

print(
    f"Duplicatas "
    f"(timestamp, scenario_id): "
    f"{duplicate_keys:,}"
)


if duplicate_rows:

    print(
        "❌ Existem linhas completamente duplicadas."
    )

else:

    print(
        "✅ Nenhuma linha completamente duplicada."
    )


if duplicate_keys:

    print(
        "❌ Existem duplicatas "
        "em (timestamp, scenario_id)."
    )

else:

    print(
        "✅ Nenhuma duplicata "
        "em (timestamp, scenario_id)."
    )


# ============================================================
# 4. CENÁRIOS
# ============================================================

print_header(
    "4. CENÁRIOS"
)


if "scenario_id" not in df.columns:

    raise ValueError(
        "Coluna scenario_id não encontrada."
    )


scenario_count = (
    df["scenario_id"].nunique()
)


print(
    f"Cenários encontrados: "
    f"{scenario_count}"
)


print(
    "\nRegistros por cenário:"
)


scenario_counts = (
    df.groupby(
        "scenario_id"
    )
    .size()
)


print(
    scenario_counts.to_string()
)


if scenario_count == 14:

    print(
        "✅ Os 14 cenários estão presentes."
    )

else:

    print(
        "⚠️ Quantidade de cenários diferente de 14."
    )


# ============================================================
# 5. VALORES AUSENTES
# ============================================================

print_header(
    "5. VALORES AUSENTES"
)


missing = (
    df.isna().sum()
)


missing_nonzero = (
    missing[
        missing > 0
    ]
)


if len(missing_nonzero) == 0:

    print(
        "✅ Nenhum valor ausente."
    )

else:

    print(
        "❌ Valores ausentes encontrados:"
    )

    print(
        missing_nonzero.to_string()
    )


# ============================================================
# 6. VALORES INFINITOS
# ============================================================

print_header(
    "6. VALORES INFINITOS"
)


numeric_cols = (
    df.select_dtypes(
        include=np.number
    ).columns
)


infinite_total = 0


for col in numeric_cols:

    count = np.isinf(
        df[col].to_numpy()
    ).sum()


    if count:

        print(
            f"❌ {col}: "
            f"{count:,}"
        )

        infinite_total += count


if infinite_total == 0:

    print(
        "✅ Nenhum valor infinito."
    )

else:

    print(
        f"❌ Total de infinitos: "
        f"{infinite_total:,}"
    )


# ============================================================
# 7. COLUNAS OBRIGATÓRIAS
# ============================================================

print_header(
    "7. COLUNAS OBRIGATÓRIAS"
)


required_cols = [

    "timestamp",
    "scenario_id",

    *HISTORICAL_FEATURES,

    *FORECAST_COLS,

    *TARGET_COLS,
]


missing_required = [
    col
    for col in required_cols
    if col not in df.columns
]


if missing_required:

    print(
        "❌ Colunas obrigatórias ausentes:"
    )

    for col in missing_required:

        print(
            f"   - {col}"
        )

else:

    print(
        f"✅ Todas as "
        f"{len(required_cols)} "
        f"colunas obrigatórias existem."
    )


# ============================================================
# 8. VERIFICAÇÃO DOS FORECASTS
# ============================================================

print_header(
    "8. VERIFICAÇÃO DOS FORECASTS"
)


forecast_missing = (
    df[FORECAST_COLS]
    .isna()
    .sum()
)


forecast_missing = (
    forecast_missing[
        forecast_missing > 0
    ]
)


if len(forecast_missing) == 0:

    print(
        "✅ Nenhum forecast possui valores ausentes."
    )

else:

    print(
        "❌ Forecasts com valores ausentes:"
    )

    print(
        forecast_missing.to_string()
    )


# ============================================================
# 9. RANGES DOS FORECASTS
# ============================================================

print_header(
    "9. RANGES DOS FORECASTS"
)


range_errors = 0


for col in FORECAST_COLS:

    values = df[col]


    if "temperature" in col:

        minimum = -10
        maximum = 60


    elif "humidity" in col:

        minimum = 0
        maximum = 100


    elif "precipitation" in col:

        minimum = 0
        maximum = 100


    elif "wind" in col:

        minimum = 0
        maximum = 100


    elif "radiation" in col:

        minimum = 0
        maximum = 1500


    elif "et0" in col:

        minimum = 0
        maximum = 20


    else:

        continue


    invalid = (
        (values < minimum)
        | (values > maximum)
    ).sum()


    if invalid:

        print(
            f"❌ {col}: "
            f"{invalid:,} fora da faixa."
        )

        range_errors += invalid

    else:

        print(
            f"✅ {col}: faixa OK."
        )


if range_errors == 0:

    print(
        "\n✅ Todos os ranges estão OK."
    )


# ============================================================
# 10. CONSISTÊNCIA FÍSICA
# ============================================================

print_header(
    "10. CONSISTÊNCIA FÍSICA"
)


physical_errors = 0


physical_rules = {

    "soil_moisture_percent": (
        0,
        100
    ),

    "kc": (
        0,
        2
    ),

    "et0": (
        0,
        20
    ),

    "etc": (
        0,
        20
    ),

    "rain_6h": (
        0,
        500
    ),

    "rain_24h": (
        0,
        1000
    ),

    "etc_6h": (
        0,
        100
    ),

    "etc_24h": (
        0,
        200
    ),

    "irrigation_depth_next_24h": (
        0,
        100
    ),
}


for col, (
    minimum,
    maximum
) in physical_rules.items():

    if col not in df.columns:

        continue


    values = df[col]


    invalid = (
        (values < minimum)
        | (values > maximum)
    ).sum()


    if invalid:

        print(
            f"❌ {col}: "
            f"{invalid:,} valores inválidos."
        )

        physical_errors += invalid

    else:

        print(
            f"✅ {col}: fisicamente plausível."
        )


# ============================================================
# 11. DISTRIBUIÇÃO DO TARGET
# ============================================================

print_header(
    "11. DISTRIBUIÇÃO DO TARGET"
)


target = df[
    "irrigation_depth_next_24h"
]


event = df[
    "irrigation_event_next_24h"
]


print(
    f"Registros: "
    f"{len(target):,}"
)


print(
    f"Eventos: "
    f"{(target > 0).sum():,}"
)


print(
    f"Não eventos: "
    f"{(target == 0).sum():,}"
)


print(
    f"Taxa de evento: "
    f"{(target > 0).mean() * 100:.4f}%"
)


print()

print(
    f"Média: "
    f"{target.mean():.4f} mm"
)

print(
    f"Mediana: "
    f"{target.median():.4f} mm"
)

print(
    f"P90: "
    f"{target.quantile(.90):.4f} mm"
)

print(
    f"P95: "
    f"{target.quantile(.95):.4f} mm"
)

print(
    f"P99: "
    f"{target.quantile(.99):.4f} mm"
)

print(
    f"Máximo: "
    f"{target.max():.4f} mm"
)


# ============================================================
# 12. CONSISTÊNCIA TARGET / EVENTO
# ============================================================

print_header(
    "12. CONSISTÊNCIA TARGET / EVENTO"
)


expected_event = (
    target > 0
).astype(int)


mismatch = (
    expected_event != event
).sum()


print(
    f"Inconsistências: "
    f"{mismatch:,}"
)


if mismatch:

    print(
        "❌ Target de evento inconsistente."
    )

else:

    print(
        "✅ Target de evento consistente."
    )


# ============================================================
# 13. EVENTOS POR CENÁRIO
# ============================================================

print_header(
    "13. EVENTOS POR CENÁRIO"
)


scenario_events = (
    df.groupby(
        "scenario_id"
    )
    .agg(
        records=(
            "timestamp",
            "count"
        ),

        events=(
            "irrigation_event_next_24h",
            "sum"
        )
    )
)


scenario_events[
    "event_rate"
] = (
    scenario_events["events"]
    / scenario_events["records"]
    * 100
)


print(
    scenario_events
    .round(4)
    .to_string()
)


# ============================================================
# 14. EVENTOS POR ANO
# ============================================================

print_header(
    "14. EVENTOS POR ANO"
)


df["year"] = (
    df["timestamp"]
    .dt.year
)


year_events = (
    df.groupby(
        "year"
    )
    .agg(
        records=(
            "timestamp",
            "count"
        ),

        events=(
            "irrigation_event_next_24h",
            "sum"
        )
    )
)


year_events[
    "event_rate"
] = (
    year_events["events"]
    / year_events["records"]
    * 100
)


print(
    year_events
    .round(4)
    .to_string()
)


# ============================================================
# 15. CORRELAÇÕES COM TARGET
# ============================================================

print_header(
    "15. CORRELAÇÕES COM TARGET"
)


correlations = (
    df[
        [
            col
            for col in numeric_cols
            if col in df.columns
        ]
    ]
    .corr()[
        "irrigation_depth_next_24h"
    ]
    .drop(
        "irrigation_depth_next_24h",
        errors="ignore"
    )
    .sort_values()
)


print(
    correlations
    .round(4)
    .to_string()
)


# ============================================================
# 16. CORRELAÇÕES ENTRE FORECASTS
# ============================================================

print_header(
    "16. CORRELAÇÕES ENTRE FORECASTS"
)


forecast_corr = (
    df[
        FORECAST_COLS
    ]
    .corr()
)


high_corr_pairs = []


for i in range(
    len(forecast_corr.columns)
):

    for j in range(
        i + 1,
        len(forecast_corr.columns)
    ):

        value = forecast_corr.iloc[
            i,
            j
        ]


        if abs(value) >= 0.95:

            high_corr_pairs.append(
                (
                    forecast_corr.columns[i],
                    forecast_corr.columns[j],
                    value
                )
            )


if high_corr_pairs:

    print(
        "Pares com |r| >= 0.95:"
    )

    for col_a, col_b, value in (
        high_corr_pairs
    ):

        print(
            f"{col_a} "
            f"<-> "
            f"{col_b}: "
            f"{value:.4f}"
        )

else:

    print(
        "Nenhum par com |r| >= 0.95."
    )


# ============================================================
# 17. VERIFICAÇÃO DE LOOK-AHEAD
# ============================================================

print_header(
    "17. VERIFICAÇÃO DE LOOK-AHEAD"
)


print(
    """
O alvo representa a necessidade de irrigação
das próximas 24 horas.

As features devem representar somente
informações disponíveis no instante t.

As colunas do próprio futuro não podem ser
usadas como observações realizadas.
"""
)


for col in [

    "irrigation_depth_next_24h",
    "irrigation_event_next_24h",

]:

    if col in df.columns:

        print(
            f"ℹ️ {col}: TARGET "
            f"(não utilizar como feature)."
        )


if "storage_after_irrigation_mm" in df.columns:

    print(
        "⚠️ storage_after_irrigation_mm: "
        "estado pós-irrigação; "
        "não utilizar como feature."
    )


# ============================================================
# 18. FEATURES SUSPEITAS DE VAZAMENTO
# ============================================================

print_header(
    "18. FEATURES SUSPEITAS DE VAZAMENTO"
)


leakage_found = []


for col in LEAKAGE_COLS:

    if col in df.columns:

        leakage_found.append(
            col
        )


if leakage_found:

    print(
        "⚠️ As seguintes colunas existem "
        "no dataset, mas NÃO devem entrar "
        "no treinamento:"
    )

    for col in leakage_found:

        print(
            f"   - {col}"
        )

else:

    print(
        "✅ Nenhuma coluna suspeita encontrada."
    )


# ============================================================
# 19. FEATURES VÁLIDAS
# ============================================================

print_header(
    "19. FEATURES VÁLIDAS PARA TREINAMENTO"
)


valid_features = []


for col in df.columns:

    if col in LEAKAGE_COLS:

        continue


    if col in TARGET_COLS:

        continue


    if col in [
        "timestamp",
        "year",
        "hour",
        "day_of_year",
    ]:

        continue


    if col in required_cols:

        valid_features.append(
            col
        )


print(
    f"Features válidas: "
    f"{len(valid_features)}"
)


for i, col in enumerate(
    valid_features,
    start=1
):

    print(
        f"{i:02d}. {col}"
    )


# ============================================================
# 20. FEATURES / TARGET
# ============================================================

print_header(
    "20. VERIFICAÇÃO FEATURES / TARGET"
)


feature_target_overlap = (
    set(valid_features)
    & set(TARGET_COLS)
)


if feature_target_overlap:

    print(
        "❌ TARGET presente nas features:"
    )

    for col in feature_target_overlap:

        print(
            f"   - {col}"
        )

else:

    print(
        "✅ Nenhum target está entre as features."
    )


# ============================================================
# 21. FEATURES / VAZAMENTO
# ============================================================

print_header(
    "21. VERIFICAÇÃO DE VAZAMENTO NAS FEATURES"
)


feature_leakage_overlap = (
    set(valid_features)
    & set(LEAKAGE_COLS)
)


if feature_leakage_overlap:

    print(
        "❌ Existem colunas de possível "
        "vazamento entre as features:"
    )

    for col in feature_leakage_overlap:

        print(
            f"   - {col}"
        )

else:

    print(
        "✅ Nenhuma coluna de vazamento "
        "está entre as features."
    )


# ============================================================
# 22. VERIFICAÇÃO TEMPORAL
# ============================================================

print_header(
    "22. VERIFICAÇÃO TEMPORAL"
)


temporal_gaps = 0


for scenario_id, group_scenario in (
    df.groupby("scenario_id")
):

    times = (
        group_scenario[
            "timestamp"
        ]
        .sort_values()
        .drop_duplicates()
        .reset_index(drop=True)
    )


    diffs = (
        times.diff()
        .dropna()
    )


    bad_mask = (
        diffs != pd.Timedelta(hours=1)
    )


    bad_count = (
        bad_mask.sum()
    )


    if bad_count == 0:

        print(
            f"✅ Cenário {scenario_id}: "
            f"sequência temporal contínua."
        )

    else:

        temporal_gaps += bad_count

        print(
            f"⚠️ Cenário {scenario_id}: "
            f"{bad_count:,} intervalo(s) "
            f"diferente(s) de 1h."
        )


        positions = np.where(
            bad_mask
        )[0]


        for position in positions:

            previous_timestamp = (
                times.iloc[position]
            )

            current_timestamp = (
                times.iloc[position + 1]
            )

            difference = (
                current_timestamp
                - previous_timestamp
            )

            print(
                f"   {previous_timestamp} "
                f"→ "
                f"{current_timestamp} "
                f"= {difference}"
            )


if temporal_gaps == 0:

    print(
        "\n✅ Nenhuma quebra temporal encontrada."
    )

else:

    print(
        f"\n⚠️ Total de intervalos "
        f"problemáticos: "
        f"{temporal_gaps:,}"
    )

    print(
        "ℹ️ As quebras temporais são "
        "mantidas como alerta porque "
        "pertencem ao histórico de origem."
    )


# ============================================================
# 23. VERIFICAÇÃO CORRETA DA CONSTRUÇÃO DO TARGET
# ============================================================

print_header(
    "23. VERIFICAÇÃO DA CONSTRUÇÃO DO TARGET"
)


target_errors = 0

target_checked = 0

target_invalid_windows = 0


for scenario_id, group_scenario in (
    df.groupby("scenario_id")
):

    group_scenario = (
        group_scenario
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


    if "irrigation_depth_mm" not in group_scenario.columns:

        print(
            f"⚠️ Cenário {scenario_id}: "
            f"irrigation_depth_mm ausente."
        )

        continue


    timestamps = (
        group_scenario[
            "timestamp"
        ]
        .to_numpy()
    )


    irrigation = (
        group_scenario[
            "irrigation_depth_mm"
        ]
        .to_numpy(
            dtype=float
        )
    )


    stored_target = (
        group_scenario[
            "irrigation_depth_next_24h"
        ]
        .to_numpy(
            dtype=float
        )
    )


    # --------------------------------------------------------
    # Mapa timestamp -> posição
    # --------------------------------------------------------

    timestamp_to_position = {
        timestamp: i
        for i, timestamp in enumerate(
            timestamps
        )
    }


    # --------------------------------------------------------
    # Verificar cada target
    # --------------------------------------------------------

    for i, timestamp in enumerate(
        timestamps
    ):

        actual = stored_target[i]


        # O gerador removeu as janelas sem 24h futuras
        # completas. Portanto, não devemos exigir target
        # para essas posições.

        target_end = (
            timestamp
            + pd.Timedelta(hours=24)
        )


        end_position = (
            timestamp_to_position.get(
                target_end
            )
        )


        # ----------------------------------------------------
        # Não existe t+24h
        # ----------------------------------------------------

        if end_position is None:

            continue


        # ----------------------------------------------------
        # Existe t+24h, mas não há exatamente
        # 24 registros entre t e t+24h.
        #
        # Isso acontece se houver uma quebra temporal.
        # ----------------------------------------------------

        if end_position != i + 24:

            target_invalid_windows += 1

            continue


        # ----------------------------------------------------
        # Janela válida
        # ----------------------------------------------------

        target_checked += 1


        expected = irrigation[
            i + 1:
            i + 25
        ].sum()


        if np.isnan(actual):

            target_errors += 1

            if target_errors <= 10:

                print(
                    f"❌ Cenário {scenario_id}, "
                    f"{timestamp}: "
                    f"target ausente em janela válida."
                )

            continue


        if not np.isclose(
            expected,
            actual,
            atol=1e-6
        ):

            target_errors += 1

            if target_errors <= 10:

                print(
                    f"❌ Cenário {scenario_id}, "
                    f"{timestamp}: "
                    f"esperado={expected:.6f}, "
                    f"encontrado={actual:.6f}"
                )


print(
    f"Janelas válidas verificadas: "
    f"{target_checked:,}"
)


print(
    f"Janelas invalidadas por quebra temporal: "
    f"{target_invalid_windows:,}"
)


print(
    f"Inconsistências encontradas: "
    f"{target_errors:,}"
)


if target_errors == 0:

    print(
        "✅ Target corresponde à soma "
        "das 24 horas futuras nas janelas "
        "temporalmente válidas."
    )

else:

    print(
        f"❌ Total de inconsistências "
        f"no target: "
        f"{target_errors:,}"
    )


# ============================================================
# 24. VERIFICAÇÃO DO EVENTO FUTURO
# ============================================================

print_header(
    "24. VERIFICAÇÃO DO EVENTO FUTURO"
)


event_errors = 0


expected_event = (
    df[
        "irrigation_depth_next_24h"
    ] > 0
).astype(int)


actual_event = (
    df[
        "irrigation_event_next_24h"
    ]
)


event_errors = (
    expected_event
    != actual_event
).sum()


if event_errors == 0:

    print(
        "✅ Evento next 24h está correto."
    )

else:

    print(
        f"❌ {event_errors:,} "
        f"inconsistências."
    )


# ============================================================
# 25. DATA DE DECISÃO
# ============================================================

print_header(
    "25. VERIFICAÇÃO DA DATA DE DECISÃO"
)


print(
    f"Primeiro instante: "
    f"{df['timestamp'].min()}"
)


print(
    f"Último instante: "
    f"{df['timestamp'].max()}"
)


print(
    """
ℹ️ Os forecasts +24h/+48h/+72h
foram previamente alinhados ao instante
em que a previsão estaria disponível.

Portanto, as colunas de forecast representam
informação prospectiva disponível em t,
enquanto o target representa a irrigação
observada/simulada entre t+1 e t+24.

Janelas que atravessam falhas temporais
não são consideradas targets válidos.
"""
)


# ============================================================
# 26. RESUMO
# ============================================================

print_header(
    "26. RESUMO DAS COLUNAS"
)


print(
    f"Total de colunas: "
    f"{len(df.columns)}"
)


print(
    f"Features válidas: "
    f"{len(valid_features)}"
)


print(
    f"Targets: "
    f"{len(TARGET_COLS)}"
)


print(
    f"Colunas de auditoria/vazamento: "
    f"{len(leakage_found)}"
)


# ============================================================
# 27. VEREDITO FINAL
# ============================================================

print_header(
    "27. VEREDITO FINAL"
)


critical_errors = 0

alerts = 0


# ------------------------------------------------------------
# ERROS CRÍTICOS
# ------------------------------------------------------------

critical_errors += (
    invalid_timestamp
)

critical_errors += (
    duplicate_rows
)

critical_errors += (
    duplicate_keys
)

critical_errors += (
    len(missing_required)
)

critical_errors += (
    len(missing_nonzero)
)

critical_errors += (
    infinite_total
)

critical_errors += (
    range_errors
)

critical_errors += (
    physical_errors
)

critical_errors += (
    mismatch
)

critical_errors += (
    target_errors
)

critical_errors += (
    event_errors
)


# ------------------------------------------------------------
# ALERTAS
# ------------------------------------------------------------

if leakage_found:

    alerts += 1


if len(high_corr_pairs):

    alerts += 1


if temporal_gaps:

    alerts += 1


# ------------------------------------------------------------
# RESULTADO
# ------------------------------------------------------------

print(
    f"Registros auditados: "
    f"{len(df):,}"
)


print(
    f"Colunas: "
    f"{len(df.columns)}"
)


print(
    f"Features válidas: "
    f"{len(valid_features)}"
)


print(
    f"Erros críticos: "
    f"{critical_errors}"
)


print(
    f"Alertas: "
    f"{alerts}"
)


if critical_errors == 0:

    print()

    print(
        "🟢 APROVADO"
    )

    print()

    print(
        "O Dataset Forecast está "
        "estruturalmente consistente "
        "e pode seguir para treinamento."
    )


    if alerts:

        print()

        print(
            "ℹ️ Existem alertas que não "
            "impedem o treinamento."
        )

        if temporal_gaps:

            print(
                "   - Existem quebras temporais "
                "herdadas do histórico de origem."
            )

        if leakage_found:

            print(
                "   - Existem colunas de auditoria/"
                "vazamento que devem permanecer "
                "fora das features."
            )

        if high_corr_pairs:

            print(
                "   - Existem forecasts "
                "altamente correlacionados."
            )


else:

    print()

    print(
        "🔴 REPROVADO"
    )

    print()

    print(
        "Existem problemas que precisam "
        "ser corrigidos antes do treinamento."
    )


print_header(
    "FIM DA AUDITORIA"
)