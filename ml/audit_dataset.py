"""
audit_dataset_3.py

Auditoria completa do Dataset 3 de irrigação.

Entrada:
    ml/data/irrigation_dataset_3.csv

Saídas:
    ml/results_3/audit_dataset_3.txt
    ml/results_3/audit_daily_etc.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

RANDOM_TOL = 1e-8
ETC_TOL = 1e-6
BALANCE_TOL = 1e-6
TIME_DIFF_EXPECTED_H = 1.0

# Mesmos fatores usados no generate_dataset_3.py
RAIN_EFFECT_MAP = {
    1: 0.70,
    2: 0.80,
    3: 0.70,
    4: 0.80,
    5: 0.75,
    6: 0.85,
    7: 0.70,
    8: 0.85,
    9: 0.75,
    10: 0.80,
    11: 0.85,
    12: 0.80,
    13: 0.85,
    14: 0.85,
}

GLOBAL_RAIN_EFFECTIVE_FRACTION = 0.75

MAX_IRRIGATION_DEPTH_MM = 15.0
RECOVERY_MARGIN_MM = 4.0

KC_BY_STAGE = {
    "initial": 0.82,
    "development": 1.03,
    "mid": 1.07,
    "late": 0.93,
}

BASE_STRESS_FRACTION = {
    "initial": 0.48,
    "development": 0.45,
    "mid": 0.43,
    "late": 0.46,
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================


def percentage(value, total):
    if total == 0:
        return 0.0

    return 100.0 * value / total


def run_audit(input_path, output_dir):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = []
    errors = []
    warnings = []

    def section(title):
        lines.append("")
        lines.append("=" * 78)
        lines.append(title)
        lines.append("=" * 78)

    def report(label, value):
        lines.append(
            f"{label:<55} {value}"
        )

    def error(message):
        errors.append(message)

        lines.append(
            f"❌ ERRO: {message}"
        )

    def warning(message):
        warnings.append(message)

        lines.append(
            f"⚠️ ALERTA: {message}"
        )

    # ========================================================
    # CABEÇALHO
    # ========================================================

    section("AUDITORIA DO DATASET 3")

    report(
        "Arquivo",
        str(input_path),
    )

    report(
        "Execução",
        pd.Timestamp.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    )

    if not input_path.exists():

        error(
            f"Arquivo não encontrado: {input_path}"
        )

        report_path = (
            output_dir
            / "audit_dataset_3.txt"
        )

        report_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        return False, report_path

    # ========================================================
    # LEITURA
    # ========================================================

    df = pd.read_csv(
        input_path,
        parse_dates=["timestamp"],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # Coluna auxiliar usada pelas auditorias temporais.
    # Não faz parte do dataset original.
    df["date"] = df["timestamp"].dt.floor("D")

    report(
        "Registros",
        f"{len(df):,}",
    )

    report(
        "Colunas",
        f"{len(df.columns):,}",
    )

    # ========================================================
    # 1. INTEGRIDADE
    # ========================================================

    section(
        "1. INTEGRIDADE E ESTRUTURA"
    )

    required_columns = [
        "id",
        "timestamp",
        "scenario_id",
        "storage_capacity_mm",
        "temperature_c",
        "air_humidity_percent",
        "wind_speed_m_s",
        "solar_radiation_kwh_m2",
        "precipitation_mm",
        "soil_moisture_percent",
        "soil_moisture_lag_1h",
        "soil_moisture_lag_3h",
        "soil_moisture_lag_6h",
        "crop_stage",
        "kc",
        "et0",
        "etc",
        "etc_6h",
        "etc_24h",
        "rain_6h",
        "rain_24h",
        "temperature_6h_mean",
        "radiation_6h_sum",
        "dynamic_lower_bound_percent",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "irrigation_depth_mm",
        "storage_before_irrigation_mm",
        "storage_after_irrigation_mm",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        error(
            f"Colunas ausentes: {missing_columns}"
        )

    else:

        report(
            "Colunas obrigatórias",
            "OK",
        )

    # --------------------------------------------------------
    # NaN
    # --------------------------------------------------------

    null_counts = df.isna().sum()

    null_columns = null_counts[
        null_counts > 0
    ]

    report(
        "Colunas com NaN",
        f"{len(null_columns):,}",
    )

    if len(null_columns):

        error(
            f"Existem valores NaN: "
            f"{null_columns.to_dict()}"
        )

    # --------------------------------------------------------
    # Duplicatas
    # --------------------------------------------------------

    duplicated_rows = int(
        df.duplicated().sum()
    )

    duplicated_ids = int(
        df["id"].duplicated().sum()
    )

    duplicated_timestamp_scenario = int(
        df.duplicated(
            ["timestamp", "scenario_id"]
        ).sum()
    )

    report(
        "Linhas duplicadas",
        f"{duplicated_rows:,}",
    )

    report(
        "IDs duplicados",
        f"{duplicated_ids:,}",
    )

    report(
        "(timestamp, scenario_id) duplicados",
        f"{duplicated_timestamp_scenario:,}",
    )

    if duplicated_rows:

        error(
            f"{duplicated_rows:,} linhas "
            "inteiras duplicadas."
        )

    if duplicated_ids:

        error(
            f"{duplicated_ids:,} IDs duplicados."
        )

    if duplicated_timestamp_scenario:

        error(
            f"{duplicated_timestamp_scenario:,} "
            "pares timestamp/scenario duplicados."
        )

    # ========================================================
    # 2. TEMPO
    # ========================================================

    section(
        "2. CONSISTÊNCIA TEMPORAL"
    )

    report(
        "Primeiro timestamp",
        str(df["timestamp"].min()),
    )

    report(
        "Último timestamp",
        str(df["timestamp"].max()),
    )

    report(
        "Timestamps únicos",
        f"{df['timestamp'].nunique():,}",
    )

    report(
        "Cenários",
        f"{df['scenario_id'].nunique():,}",
    )

    temporal_errors = 0

    for scenario_id, group in df.groupby(
        "scenario_id"
    ):

        group = group.sort_values(
            "timestamp"
        )

        differences = (
            group["timestamp"]
            .diff()
            .dropna()
            .dt.total_seconds()
            / 3600.0
        )

        invalid = (
            np.abs(
                differences
                - TIME_DIFF_EXPECTED_H
            )
            > 1e-9
        )

        count = int(
            invalid.sum()
        )

        if count:

            temporal_errors += count

            error(
                f"Cenário {scenario_id}: "
                f"{count:,} intervalos não "
                "possuem exatamente 1 hora."
            )

    if temporal_errors == 0:

        report(
            "Intervalos horários",
            "OK — todos de 1 hora",
        )

    # ========================================================
    # 3. METEOROLOGIA
    # ========================================================

    section(
        "3. FAIXAS METEOROLÓGICAS"
    )

    meteorological_columns = [
        "temperature_c",
        "air_humidity_percent",
        "wind_speed_m_s",
        "solar_radiation_kwh_m2",
        "precipitation_mm",
    ]

    for column in meteorological_columns:

        report(
            column,
            (
                f"min={df[column].min():.4f} | "
                f"média={df[column].mean():.4f} | "
                f"max={df[column].max():.4f}"
            ),
        )

    # Umidade relativa
    if (
        (df["air_humidity_percent"] < 0)
        .any()
        or
        (
            df["air_humidity_percent"]
            > 100
        ).any()
    ):

        error(
            "Umidade relativa fora do intervalo 0–100%."
        )

    # Valores que fisicamente não deveriam ser negativos
    non_negative_columns = [
        "wind_speed_m_s",
        "solar_radiation_kwh_m2",
        "precipitation_mm",
    ]

    for column in non_negative_columns:

        if (
            df[column] < -RANDOM_TOL
        ).any():

            error(
                f"{column} contém valores negativos."
            )

    # ========================================================
    # 4. Kc E ETc
    # ========================================================

    section(
        "4. AUDITORIA AGRONÔMICA E DE ETc"
    )

    # --------------------------------------------------------
    # Kc
    # --------------------------------------------------------

    kc_errors = 0

    for stage, expected_kc in KC_BY_STAGE.items():

        values = df.loc[
            df["crop_stage"] == stage,
            "kc",
        ]

        report(
            f"Kc — {stage}",
            (
                f"esperado={expected_kc:.2f} | "
                f"registros={len(values):,} | "
                f"valores únicos="
                f"{values.unique()}"
            ),
        )

        if len(values) == 0:

            error(
                f"O estágio '{stage}' "
                "não aparece no dataset."
            )

            continue

        if not np.allclose(
            values.to_numpy(),
            expected_kc,
            atol=ETC_TOL,
        ):

            kc_errors += 1

            error(
                f"Kc inconsistente no estágio "
                f"'{stage}'."
            )

    if kc_errors == 0:

        report(
            "Kc",
            "OK",
        )

    # --------------------------------------------------------
    # ET0
    # --------------------------------------------------------

    report(
        "ET0 média",
        f"{df['et0'].mean():.6f} mm/dia",
    )

    report(
        "ET0 mínimo",
        f"{df['et0'].min():.6f} mm/dia",
    )

    report(
        "ET0 máximo",
        f"{df['et0'].max():.6f} mm/dia",
    )

    if (
        df["et0"] < -RANDOM_TOL
    ).any():

        error(
            "ET0 contém valores negativos."
        )

    # --------------------------------------------------------
    # ETc horária
    # --------------------------------------------------------

    report(
        "ETc horária média",
        f"{df['etc'].mean():.8f} mm/h",
    )

    report(
        "ETc horária mínima",
        f"{df['etc'].min():.8f} mm/h",
    )

    report(
        "ETc horária máxima",
        f"{df['etc'].max():.8f} mm/h",
    )

    if (
        df["etc"] < -ETC_TOL
    ).any():

        error(
            "ETc contém valores negativos."
        )

    # --------------------------------------------------------
    # Soma diária de ETc
    # --------------------------------------------------------
    #
    # IMPORTANTE:
    # O Dataset 3 começa em 2020-01-01 23:00.
    # Portanto, o primeiro dia possui somente 1 hora
    # por cenário.
    #
    # Dias incompletos não devem ser considerados erro
    # na verificação de fechamento diário da ETc.
    # --------------------------------------------------------

    daily = (
        df
        .groupby(
            ["scenario_id", "date"],
            as_index=False,
        )
        .agg(
            etc_hourly_sum=(
                "etc",
                "sum",
            ),
            et0=(
                "et0",
                "first",
            ),
            kc=(
                "kc",
                "first",
            ),
            hours=(
                "timestamp",
                "count",
            ),
        )
    )

    daily["etc_expected"] = (
        daily["et0"]
        * daily["kc"]
    )

    daily["etc_error"] = (
        daily["etc_hourly_sum"]
        - daily["etc_expected"]
    )

    daily["abs_error"] = (
        daily["etc_error"].abs()
    )

    # Dias completos possuem exatamente 24 observações.
    complete_days = daily[
        daily["hours"] == 24
    ].copy()

    # Dias incompletos são registrados, mas não
    # entram como inconsistência de fechamento.
    partial_days = daily[
        daily["hours"] != 24
    ].copy()

    report(
        "Dias/cenários auditados",
        f"{len(complete_days):,}",
    )

    report(
        "Dias/cenários completos",
        f"{len(complete_days):,}",
    )

    report(
        "Dias/cenários parciais excluídos",
        f"{len(partial_days):,}",
    )

    if len(partial_days):

        report(
            "Motivo dos dias parciais",
            (
                "primeiro dia possui "
                "apenas observações disponíveis "
                "na entrada meteorológica"
            ),
        )

    if len(complete_days):

        report(
            "Maior erro na soma diária de ETc",
            f"{complete_days['abs_error'].max():.12f} mm",
        )

        invalid_daily = complete_days[
            complete_days["abs_error"]
            > ETC_TOL
        ]

    else:

        report(
            "Maior erro na soma diária de ETc",
            "N/A — nenhum dia completo",
        )

        invalid_daily = complete_days.copy()

    report(
        "Dias completos com inconsistência de ETc",
        f"{len(invalid_daily):,}",
    )

    if len(invalid_daily):

        error(
            f"{len(invalid_daily):,} "
            "combinações cenário/data completas "
            "falharam na auditoria da ETc diária."
        )

    daily_audit_path = (
        output_dir
        / "audit_daily_etc.csv"
    )

    daily.to_csv(
        daily_audit_path,
        index=False,
    )

    report(
        "Auditoria diária salva",
        str(daily_audit_path),
    )

    # ========================================================
    # 5. BALANÇO HÍDRICO
    # ========================================================

    section(
        "5. BALANÇO HÍDRICO"
    )

    balance_errors = []

    sorted_df = df.sort_values(
        ["scenario_id", "timestamp"]
    )

    for scenario_id, group in sorted_df.groupby(
        "scenario_id"
    ):

        group = group.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        rain_factor = RAIN_EFFECT_MAP.get(
            int(scenario_id)
        )

        if rain_factor is None:

            error(
                f"Não existe fator de chuva "
                f"para cenário {scenario_id}."
            )

            continue

        previous_storage_after = None

        for index, row in group.iterrows():

            if previous_storage_after is None:

                previous_storage_after = float(
                    row[
                        "storage_after_irrigation_mm"
                    ]
                )

                continue

            effective_rain = (
                float(row["precipitation_mm"])
                * rain_factor
                * GLOBAL_RAIN_EFFECTIVE_FRACTION
            )

            expected_before = np.clip(
                previous_storage_after
                - float(row["etc"])
                + effective_rain,
                0.0,
                float(
                    row[
                        "storage_capacity_mm"
                    ]
                ),
            )

            actual_before = float(
                row[
                    "storage_before_irrigation_mm"
                ]
            )

            balance_error = (
                actual_before
                - expected_before
            )

            if (
                abs(balance_error)
                > BALANCE_TOL
            ):

                balance_errors.append(
                    {
                        "scenario_id": scenario_id,
                        "timestamp": row[
                            "timestamp"
                        ],
                        "error_mm": balance_error,
                    }
                )

            previous_storage_after = float(
                row[
                    "storage_after_irrigation_mm"
                ]
            )

    report(
        "Erros de balanço encontrados",
        f"{len(balance_errors):,}",
    )

    if balance_errors:

        error(
            f"{len(balance_errors):,} "
            "linhas apresentam inconsistência "
            "no balanço hídrico."
        )

    else:

        report(
            "Reconstrução do balanço",
            "OK",
        )

    # --------------------------------------------------------
    # Capacidade de armazenamento
    # --------------------------------------------------------

    for column in [
        "storage_before_irrigation_mm",
        "storage_after_irrigation_mm",
    ]:

        invalid = (
            (df[column] < -RANDOM_TOL)
            |
            (
                df[column]
                >
                df["storage_capacity_mm"]
                + RANDOM_TOL
            )
        )

        count = int(
            invalid.sum()
        )

        report(
            f"{column} dentro dos limites",
            "OK"
            if count == 0
            else f"FALHOU — {count:,}",
        )

        if count:

            error(
                f"{column}: {count:,} "
                "valores fora de [0, capacidade]."
            )

    # ========================================================
    # 6. SENSOR
    # ========================================================

    section(
        "6. SENSOR DE UMIDADE"
    )

    ideal_moisture = (
        100.0
        * df[
            "storage_before_irrigation_mm"
        ]
        /
        df[
            "storage_capacity_mm"
        ]
    )

    sensor_noise = (
        df["soil_moisture_percent"]
        - ideal_moisture
    )

    report(
        "Umidade mínima",
        f"{df['soil_moisture_percent'].min():.4f}%",
    )

    report(
        "Umidade média",
        f"{df['soil_moisture_percent'].mean():.4f}%",
    )

    report(
        "Umidade máxima",
        f"{df['soil_moisture_percent'].max():.4f}%",
    )

    report(
        "Ruído médio",
        f"{sensor_noise.mean():.8f} p.p.",
    )

    report(
        "Desvio do ruído",
        f"{sensor_noise.std():.8f} p.p.",
    )

    report(
        "Ruído mínimo",
        f"{sensor_noise.min():.8f} p.p.",
    )

    report(
        "Ruído máximo",
        f"{sensor_noise.max():.8f} p.p.",
    )

    invalid_moisture = (
        (df["soil_moisture_percent"] < 0)
        |
        (df["soil_moisture_percent"] > 100)
    )

    if invalid_moisture.any():

        error(
            "Sensor de umidade possui "
            "valores fora de 0–100%."
        )

    # ========================================================
    # 7. TARGET
    # ========================================================

    section(
        "7. AUDITORIA DO TARGET"
    )

    target = df[
        "irrigation_depth_mm"
    ]

    events = df[
        target > 0
    ]

    report(
        "Total de eventos",
        f"{len(events):,}",
    )

    report(
        "Taxa de eventos",
        f"{percentage(len(events), len(df)):.4f}%",
    )

    report(
        "Target mínimo",
        f"{target.min():.8f} mm",
    )

    report(
        "Target médio",
        f"{target.mean():.8f} mm",
    )

    report(
        "Target máximo",
        f"{target.max():.8f} mm",
    )

    if len(events):

        event_depths = events[
            "irrigation_depth_mm"
        ]

        report(
            "Média por evento",
            f"{event_depths.mean():.8f} mm",
        )

        report(
            "Mediana por evento",
            f"{event_depths.median():.8f} mm",
        )

        report(
            "P90 por evento",
            f"{event_depths.quantile(0.90):.8f} mm",
        )

        report(
            "P95 por evento",
            f"{event_depths.quantile(0.95):.8f} mm",
        )

        report(
            "P99 por evento",
            f"{event_depths.quantile(0.99):.8f} mm",
        )

    if (
        target < -RANDOM_TOL
    ).any():

        error(
            "Target contém valores negativos."
        )

    if (
        target
        >
        MAX_IRRIGATION_DEPTH_MM
        + RANDOM_TOL
    ).any():

        error(
            f"Target ultrapassa "
            f"{MAX_IRRIGATION_DEPTH_MM} mm."
        )

    ceiling_events = (
        target
        >= MAX_IRRIGATION_DEPTH_MM
        - RANDOM_TOL
    )

    report(
        "Eventos no teto de 15 mm",
        f"{int(ceiling_events.sum()):,}",
    )

    # ========================================================
    # 8. DECISÃO DA IRRIGAÇÃO
    # ========================================================

    section(
        "8. AUDITORIA DA POLÍTICA DE IRRIGAÇÃO"
    )

    lower_storage = (
        df[
            "dynamic_lower_bound_percent"
        ]
        / 100.0
        *
        df[
            "storage_capacity_mm"
        ]
    )

    deficit = (
        lower_storage
        -
        df[
            "storage_before_irrigation_mm"
        ]
    ).clip(lower=0)

    expected_target = np.where(
        deficit > RANDOM_TOL,
        np.minimum(
            deficit + RECOVERY_MARGIN_MM,
            MAX_IRRIGATION_DEPTH_MM,
        ),
        0.0,
    )

    target_error = (
        target
        -
        expected_target
    ).abs()

    report(
        "Maior erro target × política",
        f"{target_error.max():.12f} mm",
    )

    if (
        target_error
        > RANDOM_TOL
    ).any():

        error(
            "Existem linhas em que "
            "irrigation_depth_mm não corresponde "
            "à política implementada."
        )

    positive_without_deficit = (
        (target > 0)
        &
        (deficit <= RANDOM_TOL)
    )

    zero_with_deficit = (
        (target == 0)
        &
        (deficit > RANDOM_TOL)
    )

    report(
        "Eventos sem déficit",
        f"{int(positive_without_deficit.sum()):,}",
    )

    report(
        "Linhas com déficit sem irrigação",
        f"{int(zero_with_deficit.sum()):,}",
    )

    if positive_without_deficit.any():

        error(
            "Há eventos de irrigação sem "
            "déficit segundo a política."
        )

    if zero_with_deficit.any():

        error(
            "Há linhas com déficit segundo "
            "a política mas sem irrigação."
        )

    # ========================================================
    # 9. PRÉ → PÓS IRRIGAÇÃO
    # ========================================================

    section(
        "9. TRANSIÇÃO PRÉ → PÓS IRRIGAÇÃO"
    )

    expected_after = np.minimum(
        df[
            "storage_before_irrigation_mm"
        ]
        + target,
        df[
            "storage_capacity_mm"
        ],
    )

    after_error = (
        df[
            "storage_after_irrigation_mm"
        ]
        -
        expected_after
    ).abs()

    report(
        "Maior erro storage_after",
        f"{after_error.max():.12f} mm",
    )

    report(
        "Linhas inconsistentes",
        f"{int((after_error > BALANCE_TOL).sum()):,}",
    )

    if (
        after_error
        > BALANCE_TOL
    ).any():

        error(
            "storage_after_irrigation não "
            "corresponde ao estado pré + irrigação."
        )

    # ========================================================
    # 10. FEATURES DE MEMÓRIA
    # ========================================================

    section(
        "10. FEATURES DE MEMÓRIA"
    )

    memory_columns = [
        "soil_moisture_lag_1h",
        "soil_moisture_lag_3h",
        "soil_moisture_lag_6h",
        "etc_6h",
        "etc_24h",
        "rain_6h",
        "rain_24h",
        "temperature_6h_mean",
        "radiation_6h_sum",
    ]

    for column in memory_columns:

        null_count = int(
            df[column].isna().sum()
        )

        report(
            f"{column} — NaN",
            f"{null_count:,}",
        )

        if null_count == len(df):

            error(
                f"{column} está 100% NaN."
            )

        elif null_count:

            warning(
                f"{column} possui "
                f"{null_count:,} NaN."
            )

    # ========================================================
    # 11. DATA LEAKAGE
    # ========================================================

    section(
        "11. AUDITORIA DE DATA LEAKAGE"
    )

    leakage_columns = [
        "irrigation_depth_mm",
        "storage_before_irrigation_mm",
        "storage_after_irrigation_mm",
        "dynamic_lower_bound_percent",
        "scenario_id",
    ]

    for column in leakage_columns:

        if column in df.columns:

            report(
                column,
                "PRESENTE — NÃO USAR COMO FEATURE",
            )

    if "storage_capacity_mm" in df.columns:

        report(
            "storage_capacity_mm",
            (
                "PRESENTE — usar somente se "
                "for configuração conhecida "
                "no momento da previsão"
            ),
        )

    # ========================================================
    # 12. DISTRIBUIÇÃO DOS EVENTOS
    # ========================================================

    section(
        "12. DISTRIBUIÇÃO DOS EVENTOS"
    )

    bins = [
        0,
        4,
        5,
        6,
        7,
        8,
        10,
        12,
        15,
        np.inf,
    ]

    labels = [
        "0–4",
        "4–5",
        "5–6",
        "6–7",
        "7–8",
        "8–10",
        "10–12",
        "12–15",
        ">15",
    ]

    distribution = pd.cut(
        events[
            "irrigation_depth_mm"
        ],
        bins=bins,
        labels=labels,
        right=False,
    ).value_counts(
        sort=False
    )

    for label, count in distribution.items():

        report(
            f"Eventos {label} mm",
            (
                f"{count:,} "
                f"({percentage(count, len(events)):.2f}%)"
            ),
        )

    # ========================================================
    # 13. EVENTOS POR CENÁRIO
    # ========================================================

    section(
        "13. EVENTOS POR CENÁRIO"
    )

    scenario_events = (
        events
        .groupby(
            "scenario_id"
        )
        .size()
    )

    lines.append(
        scenario_events.to_string()
    )

    # ========================================================
    # 14. EVENTOS POR ESTÁGIO
    # ========================================================

    section(
        "14. EVENTOS POR ESTÁGIO"
    )

    stage_events = (
        events
        .groupby(
            "crop_stage"
        )
        .size()
    )

    lines.append(
        stage_events.to_string()
    )

    # ========================================================
    # 15. CORRELAÇÕES
    # ========================================================

    section(
        "15. CORRELAÇÕES COM O TARGET"
    )

    correlation_columns = [
        "temperature_c",
        "air_humidity_percent",
        "wind_speed_m_s",
        "solar_radiation_kwh_m2",
        "precipitation_mm",
        "soil_moisture_percent",
        "kc",
        "et0",
        "etc",
        "etc_6h",
        "etc_24h",
        "rain_6h",
        "rain_24h",
        "irrigation_depth_mm",
    ]

    available_columns = [
        column
        for column in correlation_columns
        if column in df.columns
    ]

    correlations = (
        df[
            available_columns
        ]
        .corr(
            numeric_only=True
        )[
            "irrigation_depth_mm"
        ]
        .sort_values()
    )

    lines.append(
        correlations.to_string()
    )

    # ========================================================
    # 16. RELAÇÃO UMIDADE × EVENTO
    # ========================================================

    section(
        "16. RELAÇÃO UMIDADE × IRRIGAÇÃO"
    )

    median_moisture = (
        df[
            "soil_moisture_percent"
        ].median()
    )

    low_moisture = df[
        df[
            "soil_moisture_percent"
        ]
        <
        median_moisture
    ]

    high_moisture = df[
        df[
            "soil_moisture_percent"
        ]
        >=
        median_moisture
    ]

    low_event_rate = percentage(
        (
            low_moisture[
                "irrigation_depth_mm"
            ]
            > 0
        ).sum(),
        len(low_moisture),
    )

    high_event_rate = percentage(
        (
            high_moisture[
                "irrigation_depth_mm"
            ]
            > 0
        ).sum(),
        len(high_moisture),
    )

    report(
        "Mediana da umidade",
        f"{median_moisture:.4f}%",
    )

    report(
        "Taxa de evento abaixo da mediana",
        f"{low_event_rate:.4f}%",
    )

    report(
        "Taxa de evento acima/acima da mediana",
        f"{high_event_rate:.4f}%",
    )

    # ========================================================
    # 17. RESUMO DO DATASET
    # ========================================================

    section(
        "17. RESUMO"
    )

    report(
        "Registros",
        f"{len(df):,}",
    )

    report(
        "Cenários",
        f"{df['scenario_id'].nunique():,}",
    )

    report(
        "Eventos",
        f"{len(events):,}",
    )

    report(
        "Taxa de eventos",
        f"{percentage(len(events), len(df)):.4f}%",
    )

    report(
        "Água total",
        f"{target.sum():.4f} mm",
    )

    report(
        "Umidade média",
        f"{df['soil_moisture_percent'].mean():.4f}%",
    )

    report(
        "ET0 média",
        f"{df['et0'].mean():.6f} mm/dia",
    )

    report(
        "ETc média",
        f"{df['etc'].mean():.8f} mm/h",
    )

    # ========================================================
    # 18. VEREDITO
    # ========================================================

    section(
        "18. VEREDITO"
    )

    if not errors:

        lines.append(
            "✅ APROVADO — nenhuma "
            "inconsistência crítica encontrada."
        )

    else:

        lines.append(
            "❌ REPROVADO — "
            f"{len(errors)} inconsistência(s) "
            "crítica(s) encontrada(s)."
        )

    report(
        "Quantidade de erros",
        str(len(errors)),
    )

    report(
        "Quantidade de alertas",
        str(len(warnings)),
    )

    if warnings:

        lines.append("")
        lines.append("ALERTAS:")

        for message in warnings:

            lines.append(
                f"- {message}"
            )

    if errors:

        lines.append("")
        lines.append("ERROS:")

        for message in errors:

            lines.append(
                f"- {message}"
            )

    # ========================================================
    # SALVAR RELATÓRIO
    # ========================================================

    report_path = (
        output_dir
        / "audit_dataset_3.txt"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return (
        len(errors) == 0,
        report_path,
    )


# ============================================================
# MAIN
# ============================================================


def main():

    root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    input_path = (
        root
        / "ml"
        / "data"
        / "irrigation_dataset_3.csv"
    )

    output_dir = (
        root
        / "ml"
        / "results_3"
    )

    print("=" * 78)
    print("AUDITORIA DO DATASET 3")
    print("=" * 78)

    print()
    print(
        "Arquivo:",
        input_path,
    )

    ok, report_path = run_audit(
        input_path,
        output_dir,
    )

    print()
    print("=" * 78)

    if ok:

        print(
            "✅ AUDITORIA APROVADA"
        )

    else:

        print(
            "❌ AUDITORIA REPROVADA"
        )

    print("=" * 78)

    print()
    print(
        "Relatório salvo em:"
    )

    print(
        report_path
    )

    print()


if __name__ == "__main__":
    main()