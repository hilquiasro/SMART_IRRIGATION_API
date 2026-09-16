from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "irrigation_dataset_forecast_current.csv"
)


# ============================================================
# FEATURES
# ============================================================

HISTORICAL_FEATURES = [
    "soil_moisture_percent",
    "soil_moisture_lag_1h",
    "soil_moisture_lag_3h",
    "soil_moisture_lag_6h",
    "kc",
    "et0",
    "etc",
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


FORECAST_FEATURES = [
    "forecast_temperature_24h_c",
    "forecast_humidity_24h_percent",
    "forecast_precipitation_24h_mm",
    "forecast_wind_24h_m_s",
    "forecast_radiation_24h_w_m2",
    "forecast_et0_24h_mm",

    "forecast_temperature_48h_c",
    "forecast_humidity_48h_percent",
    "forecast_precipitation_48h_mm",
    "forecast_wind_48h_m_s",
    "forecast_radiation_48h_w_m2",
    "forecast_et0_48h_mm",

    "forecast_temperature_72h_c",
    "forecast_humidity_72h_percent",
    "forecast_precipitation_72h_mm",
    "forecast_wind_72h_m_s",
    "forecast_radiation_72h_w_m2",
    "forecast_et0_72h_mm",

    "forecast_rain_total_72h",
    "forecast_temperature_mean_72h",
    "forecast_humidity_mean_72h",
    "forecast_wind_mean_72h",
    "forecast_radiation_mean_72h",
    "forecast_et0_mean_72h",
]


TARGET_COLUMNS = [
    "irrigation_depth_current",
    "irrigation_event_current",
]


# ============================================================
# RESULTADOS DA AUDITORIA
# ============================================================

audit_results = {}


# ============================================================
# CARREGAR DATASET
# ============================================================

def load_dataset():

    print("=" * 70)
    print("AUDITORIA DO DATASET FORECAST-CURRENT")
    print("=" * 70)

    print("\nArquivo:")
    print(DATASET_PATH)

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{DATASET_PATH}"
        )

    print("\nCarregando dataset...")

    df = pd.read_csv(DATASET_PATH)

    print(
        f"Registros carregados: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# 1. ESTRUTURA
# ============================================================

def check_structure(df):

    print("\n" + "=" * 70)
    print("1. ESTRUTURA DO DATASET")
    print("=" * 70)

    expected_columns = [
        "timestamp",
        "scenario_id",
        *HISTORICAL_FEATURES,
        *FORECAST_FEATURES,
        *TARGET_COLUMNS,
    ]

    missing = [
        col
        for col in expected_columns
        if col not in df.columns
    ]

    additional = [
        col
        for col in df.columns
        if col not in expected_columns
    ]

    print(
        f"Colunas esperadas encontradas: "
        f"{len(expected_columns) - len(missing)}"
    )

    print(
        f"Colunas existentes no arquivo: "
        f"{len(df.columns)}"
    )

    if missing:

        print("\nCOLUNAS AUSENTES:")

        for col in missing:
            print(f"  - {col}")

        audit_results["structure"] = False

    else:

        print("Todas as colunas esperadas estão presentes.")

        audit_results["structure"] = True

    if additional:

        print("\nColunas adicionais:")

        for col in additional:
            print(f"  - {col}")

    else:

        print("Nenhuma coluna adicional.")

    if missing:

        audit_results["structure"] = False


# ============================================================
# 2. DUPLICIDADES
# ============================================================

def check_duplicates(df):

    print("\n" + "=" * 70)
    print("2. DUPLICIDADES")
    print("=" * 70)

    # --------------------------------------------------------
    # Linhas totalmente duplicadas
    # --------------------------------------------------------

    full_duplicates = df.duplicated().sum()

    print(
        f"Linhas totalmente duplicadas: "
        f"{full_duplicates:,}"
    )

    # --------------------------------------------------------
    # Timestamps duplicados
    #
    # Isso é esperado porque existem 14 cenários.
    # --------------------------------------------------------

    timestamp_duplicates = (
        df.duplicated(
            subset=["timestamp"]
        ).sum()
    )

    print(
        f"Timestamps duplicados: "
        f"{timestamp_duplicates:,}"
    )

    if timestamp_duplicates > 0:

        print(
            "Isso é esperado porque existem "
            "múltiplos cenários."
        )

    # --------------------------------------------------------
    # Chave correta
    # --------------------------------------------------------

    key_duplicates = (
        df.duplicated(
            subset=[
                "timestamp",
                "scenario_id",
            ]
        ).sum()
    )

    print(
        f"Duplicidades em "
        f"(timestamp, scenario_id): "
        f"{key_duplicates:,}"
    )

    if (
        full_duplicates == 0
        and key_duplicates == 0
    ):

        print("Chave (timestamp, scenario_id): OK")

        audit_results["duplicates"] = True

    else:

        print("PROBLEMA DE DUPLICIDADE ENCONTRADO.")

        audit_results["duplicates"] = False


# ============================================================
# 3. VALORES AUSENTES
# ============================================================

def check_missing(df):

    print("\n" + "=" * 70)
    print("3. VALORES AUSENTES")
    print("=" * 70)

    total_nan = df.isna().sum().sum()

    print(
        f"Total de valores NaN: "
        f"{total_nan:,}"
    )

    if total_nan == 0:

        print("Valores ausentes: 0")

        audit_results["missing"] = True

    else:

        print("\nColunas com NaN:")

        missing = (
            df.isna()
            .sum()
        )

        missing = missing[
            missing > 0
        ]

        print(missing)

        audit_results["missing"] = False


# ============================================================
# 4. TIMESTAMPS
# ============================================================

def check_timestamps(df):

    print("\n" + "=" * 70)
    print("4. TIMESTAMPS")
    print("=" * 70)

    try:

        timestamps = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

    except Exception as exc:

        print(
            f"Erro ao converter timestamps: {exc}"
        )

        audit_results["timestamps"] = False
        return

    invalid = timestamps.isna().sum()

    if invalid > 0:

        print(
            f"Timestamps inválidos: "
            f"{invalid:,}"
        )

        audit_results["timestamps"] = False
        return

    # --------------------------------------------------------
    # Trabalhamos com timestamps únicos.
    #
    # Os 14 cenários fazem o mesmo timestamp aparecer
    # 14 vezes no dataset.
    # --------------------------------------------------------

    unique_timestamps = (
        timestamps
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    print(
        f"Timestamps únicos: "
        f"{len(unique_timestamps):,}"
    )

    print(
        f"Primeiro timestamp: "
        f"{unique_timestamps.iloc[0]}"
    )

    print(
        f"Último timestamp: "
        f"{unique_timestamps.iloc[-1]}"
    )

    # --------------------------------------------------------
    # Intervalos entre timestamps únicos
    # --------------------------------------------------------

    differences = (
        unique_timestamps
        .diff()
        .dropna()
    )

    expected_interval = pd.Timedelta(hours=1)

    invalid_intervals = (
        differences != expected_interval
    )

    invalid_count = invalid_intervals.sum()

    print(
        f"\nIntervalos diferentes de 1h: "
        f"{invalid_count:,}"
    )

    if invalid_count == 0:

        print("Sequência temporal: OK")

        audit_results["timestamps"] = True

    else:

        print(
            "\nIntervalos problemáticos:"
        )

        problematic = differences[
            invalid_intervals
        ]

        print(problematic.head(20))

        audit_results["timestamps"] = False


# ============================================================
# 5. CENÁRIOS
# ============================================================

def check_scenarios(df):

    print("\n" + "=" * 70)
    print("5. CENÁRIOS")
    print("=" * 70)

    scenarios = sorted(
        df["scenario_id"]
        .dropna()
        .unique()
    )

    print(
        f"Número de cenários: "
        f"{len(scenarios)}"
    )

    print(
        f"Cenários: "
        f"{scenarios}"
    )

    records_per_scenario = (
        df.groupby("scenario_id")
        .size()
    )

    print("\nRegistros por cenário:")

    print(records_per_scenario)

    if len(scenarios) == 0:

        audit_results["scenarios"] = False
        return

    expected = records_per_scenario.iloc[0]

    if (
        records_per_scenario == expected
    ).all():

        print(
            "Quantidade de registros "
            "por cenário: OK"
        )

        audit_results["scenarios"] = True

    else:

        print(
            "Quantidade de registros "
            "por cenário: FALHA"
        )

        audit_results["scenarios"] = False


# ============================================================
# 6. FEATURES HISTÓRICAS
# ============================================================

def check_historical_features(df):

    print("\n" + "=" * 70)
    print("6. FEATURES HISTÓRICAS")
    print("=" * 70)

    missing = [
        col
        for col in HISTORICAL_FEATURES
        if col not in df.columns
    ]

    if missing:

        print("Features ausentes:")

        for col in missing:
            print(f"  - {col}")

        audit_results["historical_features"] = False
        return

    print(
        f"Features históricas verificadas: "
        f"{len(HISTORICAL_FEATURES)}"
    )

    non_numeric = [
        col
        for col in HISTORICAL_FEATURES
        if not pd.api.types.is_numeric_dtype(
            df[col]
        )
    ]

    if non_numeric:

        print("Features não numéricas:")

        for col in non_numeric:
            print(f"  - {col}")

        audit_results["historical_features"] = False

    else:

        print("Tipo numérico: OK")

        audit_results["historical_features"] = True


# ============================================================
# 7. FEATURES DE FORECAST
# ============================================================

def check_forecast_features(df):

    print("\n" + "=" * 70)
    print("7. FEATURES DE PREVISÃO")
    print("=" * 70)

    missing = [
        col
        for col in FORECAST_FEATURES
        if col not in df.columns
    ]

    if missing:

        print("Features ausentes:")

        for col in missing:
            print(f"  - {col}")

        audit_results["forecast_features"] = False
        return

    print(
        f"Features de forecast verificadas: "
        f"{len(FORECAST_FEATURES)}"
    )

    non_numeric = [
        col
        for col in FORECAST_FEATURES
        if not pd.api.types.is_numeric_dtype(
            df[col]
        )
    ]

    if non_numeric:

        print("Features não numéricas:")

        for col in non_numeric:
            print(f"  - {col}")

        audit_results["forecast_features"] = False
        return

    print("Tipo numérico: OK")

    nan_forecast = (
        df[FORECAST_FEATURES]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"NaN nas previsões: "
        f"{nan_forecast:,}"
    )

    if nan_forecast == 0:

        audit_results["forecast_features"] = True

    else:

        audit_results["forecast_features"] = False

    # --------------------------------------------------------
    # Resumo
    # --------------------------------------------------------

    summary_columns = [
        "forecast_temperature_24h_c",
        "forecast_humidity_24h_percent",
        "forecast_precipitation_24h_mm",
        "forecast_wind_24h_m_s",
        "forecast_radiation_24h_w_m2",
        "forecast_et0_24h_mm",
        "forecast_temperature_48h_c",
        "forecast_precipitation_48h_mm",
        "forecast_temperature_72h_c",
        "forecast_precipitation_72h_mm",
        "forecast_rain_total_72h",
        "forecast_et0_mean_72h",
    ]

    print(
        "\nResumo das principais previsões:"
    )

    print(
        df[summary_columns]
        .describe()
        .to_string()
    )


# ============================================================
# 8. PLAUSIBILIDADE DO FORECAST
# ============================================================

def check_forecast_ranges(df):

    print("\n" + "=" * 70)
    print("8. PLAUSIBILIDADE DAS PREVISÕES")
    print("=" * 70)

    ranges = {

        "temperature": (
            [
                "forecast_temperature_24h_c",
                "forecast_temperature_48h_c",
                "forecast_temperature_72h_c",
            ],
            -10,
            60,
            "°C",
        ),

        "humidity": (
            [
                "forecast_humidity_24h_percent",
                "forecast_humidity_48h_percent",
                "forecast_humidity_72h_percent",
            ],
            0,
            100,
            "%",
        ),

        "precipitation": (
            [
                "forecast_precipitation_24h_mm",
                "forecast_precipitation_48h_mm",
                "forecast_precipitation_72h_mm",
                "forecast_rain_total_72h",
            ],
            0,
            np.inf,
            "mm",
        ),

        "wind": (
            [
                "forecast_wind_24h_m_s",
                "forecast_wind_48h_m_s",
                "forecast_wind_72h_m_s",
            ],
            0,
            100,
            "m/s",
        ),

        "radiation": (
            [
                "forecast_radiation_24h_w_m2",
                "forecast_radiation_48h_w_m2",
                "forecast_radiation_72h_w_m2",
                "forecast_radiation_mean_72h",
            ],
            0,
            np.inf,
            "W/m²",
        ),

        "et0": (
            [
                "forecast_et0_24h_mm",
                "forecast_et0_48h_mm",
                "forecast_et0_72h_mm",
                "forecast_et0_mean_72h",
            ],
            0,
            np.inf,
            "mm",
        ),
    }

    failed = False

    for _, (
        columns,
        minimum,
        maximum,
        unit,
    ) in ranges.items():

        for col in columns:

            if col not in df.columns:
                continue

            min_value = df[col].min()
            max_value = df[col].max()

            print(
                f"{col}: "
                f"{min_value:.4f} → "
                f"{max_value:.4f} {unit}"
            )

            if min_value < minimum:

                print(
                    f"  PROBLEMA: valor abaixo "
                    f"do mínimo permitido."
                )

                failed = True

            if (
                np.isfinite(maximum)
                and max_value > maximum
            ):

                print(
                    f"  PROBLEMA: valor acima "
                    f"do máximo permitido."
                )

                failed = True

    if failed:

        print(
            "\nPROBLEMAS ENCONTRADOS."
        )

        audit_results["forecast_ranges"] = False

    else:

        print(
            "\nFaixas meteorológicas: OK"
        )

        audit_results["forecast_ranges"] = True


# ============================================================
# 9. TARGET ATUAL
# ============================================================

def check_current_target(df):

    print("\n" + "=" * 70)
    print("9. TARGET ATUAL")
    print("=" * 70)

    depth = df[
        "irrigation_depth_current"
    ]

    event = df[
        "irrigation_event_current"
    ]

    print(
        f"Registros: "
        f"{len(df):,}"
    )

    events = (
        event == 1
    ).sum()

    non_events = (
        event == 0
    ).sum()

    print(
        f"Eventos: "
        f"{events:,}"
    )

    print(
        f"Não eventos: "
        f"{non_events:,}"
    )

    print(
        f"Taxa de eventos: "
        f"{event.mean() * 100:.4f}%"
    )

    print(
        "\nDistribuição da lâmina atual:"
    )

    print(
        depth.describe()
    )

    negative = (
        depth < 0
    ).sum()

    print(
        f"\nLâminas negativas: "
        f"{negative:,}"
    )

    inconsistent = (
        (
            depth > 0
        ).astype(int)
        != event
    ).sum()

    print(
        f"Inconsistências "
        f"evento/lâmina: "
        f"{inconsistent:,}"
    )

    zero = (
        depth == 0
    ).sum()

    print(
        f"Lâmina igual a zero: "
        f"{zero:,}"
    )

    if (
        negative == 0
        and inconsistent == 0
    ):

        print(
            "\nConsistência do target: OK"
        )

        audit_results["target"] = True

    else:

        print(
            "\nConsistência do target: FALHA"
        )

        audit_results["target"] = False


# ============================================================
# 10. UMIDADE DO SOLO
# ============================================================

def check_soil_moisture(df):

    print("\n" + "=" * 70)
    print("10. UMIDADE DO SOLO")
    print("=" * 70)

    moisture = df[
        "soil_moisture_percent"
    ]

    print(
        f"Mínimo: "
        f"{moisture.min():.4f}%"
    )

    print(
        f"Média: "
        f"{moisture.mean():.4f}%"
    )

    print(
        f"Mediana: "
        f"{moisture.median():.4f}%"
    )

    print(
        f"Máximo: "
        f"{moisture.max():.4f}%"
    )

    outside = (
        (moisture < 0)
        | (moisture > 100)
    ).sum()

    print(
        f"\nValores fora de 0–100%: "
        f"{outside:,}"
    )

    if outside == 0:

        print(
            "Faixa da umidade do solo: OK"
        )

        audit_results["soil_moisture"] = True

    else:

        print(
            "Faixa da umidade do solo: FALHA"
        )

        audit_results["soil_moisture"] = False


# ============================================================
# 11. Kc
# ============================================================

def check_kc(df):

    print("\n" + "=" * 70)
    print("11. Kc DO COENTRO")
    print("=" * 70)

    if "kc" not in df.columns:

        print(
            "Coluna kc não está no dataset final."
        )

        audit_results["kc"] = False
        return

    valid_kc = {
        0.82,
        1.03,
        1.07,
        0.93,
    }

    unique_kc = (
        df["kc"]
        .dropna()
        .unique()
    )

    invalid = [
        value
        for value in unique_kc
        if not any(
            np.isclose(
                value,
                expected,
                atol=1e-6
            )
            for expected in valid_kc
        )
    ]

    print(
        "Valores Kc encontrados:"
    )

    print(
        sorted(
            float(x)
            for x in unique_kc
        )
    )

    if invalid:

        print(
            "Valores Kc inválidos:"
        )

        print(invalid)

        audit_results["kc"] = False

    else:

        print(
            "Valores Kc compatíveis "
            "com o coentro: OK"
        )

        audit_results["kc"] = True


# ============================================================
# 12. CONSISTÊNCIA DAS AGREGAÇÕES
# ============================================================

def check_forecast_aggregates(df):

    print("\n" + "=" * 70)
    print("12. CONSISTÊNCIA DAS AGREGAÇÕES")
    print("=" * 70)

    # --------------------------------------------------------
    # Chuva
    # --------------------------------------------------------

    expected_rain = (
        df["forecast_precipitation_24h_mm"]
        + df["forecast_precipitation_48h_mm"]
        + df["forecast_precipitation_72h_mm"]
    )

    rain_error = np.abs(
        expected_rain
        - df["forecast_rain_total_72h"]
    ).max()

    print(
        f"Erro máximo na chuva "
        f"total 72h: "
        f"{rain_error:.12f}"
    )

    # --------------------------------------------------------
    # Temperatura
    # --------------------------------------------------------

    expected_temp = df[
        [
            "forecast_temperature_24h_c",
            "forecast_temperature_48h_c",
            "forecast_temperature_72h_c",
        ]
    ].mean(axis=1)

    temp_error = np.abs(
        expected_temp
        - df["forecast_temperature_mean_72h"]
    ).max()

    print(
        f"Erro máximo na temperatura "
        f"média: "
        f"{temp_error:.12f}"
    )

    # --------------------------------------------------------
    # Umidade
    # --------------------------------------------------------

    expected_humidity = df[
        [
            "forecast_humidity_24h_percent",
            "forecast_humidity_48h_percent",
            "forecast_humidity_72h_percent",
        ]
    ].mean(axis=1)

    humidity_error = np.abs(
        expected_humidity
        - df["forecast_humidity_mean_72h"]
    ).max()

    print(
        f"Erro máximo na umidade "
        f"média: "
        f"{humidity_error:.12f}"
    )

    # --------------------------------------------------------
    # Vento
    # --------------------------------------------------------

    expected_wind = df[
        [
            "forecast_wind_24h_m_s",
            "forecast_wind_48h_m_s",
            "forecast_wind_72h_m_s",
        ]
    ].mean(axis=1)

    wind_error = np.abs(
        expected_wind
        - df["forecast_wind_mean_72h"]
    ).max()

    print(
        f"Erro máximo no vento "
        f"médio: "
        f"{wind_error:.12f}"
    )

    # --------------------------------------------------------
    # Radiação
    # --------------------------------------------------------

    expected_radiation = df[
        [
            "forecast_radiation_24h_w_m2",
            "forecast_radiation_48h_w_m2",
            "forecast_radiation_72h_w_m2",
        ]
    ].mean(axis=1)

    radiation_error = np.abs(
        expected_radiation
        - df["forecast_radiation_mean_72h"]
    ).max()

    print(
        f"Erro máximo na radiação "
        f"média: "
        f"{radiation_error:.12f}"
    )

    # --------------------------------------------------------
    # ET0
    # --------------------------------------------------------

    expected_et0 = df[
        [
            "forecast_et0_24h_mm",
            "forecast_et0_48h_mm",
            "forecast_et0_72h_mm",
        ]
    ].mean(axis=1)

    et0_error = np.abs(
        expected_et0
        - df["forecast_et0_mean_72h"]
    ).max()

    print(
        f"Erro máximo na ET0 "
        f"média: "
        f"{et0_error:.12f}"
    )

    max_error = max(
        rain_error,
        temp_error,
        humidity_error,
        wind_error,
        radiation_error,
        et0_error,
    )

    if max_error <= 1e-9:

        print(
            "\nAgregações do forecast: OK"
        )

        audit_results[
            "forecast_aggregates"
        ] = True

    else:

        print(
            "\nAgregações do forecast: FALHA"
        )

        audit_results[
            "forecast_aggregates"
        ] = False


# ============================================================
# 13. COMPLETUDE DOS CENÁRIOS
# ============================================================

def check_scenario_completeness(df):

    print("\n" + "=" * 70)
    print("13. COMPLETUDE DOS CENÁRIOS")
    print("=" * 70)

    timestamps = (
        df["timestamp"]
        .nunique()
    )

    scenarios = (
        df["scenario_id"]
        .nunique()
    )

    expected_records = (
        timestamps * scenarios
    )

    actual_records = len(df)

    print(
        f"Timestamps únicos: "
        f"{timestamps:,}"
    )

    print(
        f"Cenários: "
        f"{scenarios}"
    )

    print(
        f"Registros esperados: "
        f"{expected_records:,}"
    )

    print(
        f"Registros encontrados: "
        f"{actual_records:,}"
    )

    if actual_records == expected_records:

        print(
            "\nTodos os timestamps possuem "
            "todos os cenários."
        )

        audit_results[
            "scenario_completeness"
        ] = True

    else:

        print(
            "\nCOMPLETUDE DOS CENÁRIOS: FALHA"
        )

        audit_results[
            "scenario_completeness"
        ] = False


# ============================================================
# 14. CORRELAÇÕES
# ============================================================

def check_correlations(df):

    print("\n" + "=" * 70)
    print("14. CORRELAÇÕES COM O TARGET")
    print("=" * 70)

    target = "irrigation_depth_current"

    correlation_features = (
        HISTORICAL_FEATURES
        + FORECAST_FEATURES
    )

    correlations = (
        df[
            correlation_features
            + [target]
        ]
        .corr(numeric_only=True)[target]
        .drop(target)
        .sort_values()
    )

    print(
        correlations.to_string()
    )


# ============================================================
# 15. RESULTADO FINAL
# ============================================================

def final_result(df):

    print("\n" + "=" * 70)
    print("15. RESULTADO FINAL DA AUDITORIA")
    print("=" * 70)

    print(
        f"Registros auditados: "
        f"{len(df):,}"
    )

    print(
        f"Colunas auditadas: "
        f"{len(df.columns)}"
    )

    critical_checks = [
        "structure",
        "duplicates",
        "missing",
        "timestamps",
        "historical_features",
        "forecast_features",
        "forecast_ranges",
        "target",
        "soil_moisture",
        "kc",
        "forecast_aggregates",
        "scenario_completeness",
    ]

    # --------------------------------------------------------
    # Exibir status individual
    # --------------------------------------------------------

    print("\nStatus das verificações:")

    for check in critical_checks:

        status = audit_results.get(
            check,
            False
        )

        print(
            f"  {check:<25} "
            f"{'OK' if status else 'FALHA'}"
        )

    # --------------------------------------------------------
    # APROVAÇÃO
    # --------------------------------------------------------

    failed_checks = [
        check
        for check in critical_checks
        if not audit_results.get(
            check,
            False
        )
    ]

    if not failed_checks:

        print(
            "\nSTATUS: APROVADO"
        )

        print(
            "\nNenhuma inconsistência crítica "
            "foi encontrada nas verificações "
            "realizadas."
        )

    else:

        print(
            "\nSTATUS: REPROVADO / "
            "NECESSITA CORREÇÃO"
        )

        print(
            "\nProblemas críticos:"
        )

        for check in failed_checks:

            print(
                f"  - {check}"
            )

    print("\nResumo:")

    for check in critical_checks:

        status = audit_results.get(
            check,
            False
        )

        print(
            f"  {check:<25} "
            f"{'OK' if status else 'FALHA'}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    check_structure(df)

    check_duplicates(df)

    check_missing(df)

    check_timestamps(df)

    check_scenarios(df)

    check_historical_features(df)

    check_forecast_features(df)

    check_forecast_ranges(df)

    check_current_target(df)

    check_soil_moisture(df)

    check_kc(df)

    check_forecast_aggregates(df)

    check_scenario_completeness(df)

    check_correlations(df)

    final_result(df)

    print("\n" + "=" * 70)
    print("AUDITORIA FINALIZADA")
    print("=" * 70)


if __name__ == "__main__":
    main()