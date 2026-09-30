# from pathlib import Path
# from itertools import product

# import joblib
# import numpy as np
# import pandas as pd

# from sklearn.ensemble import (
#     ExtraTreesClassifier,
#     ExtraTreesRegressor,
#     HistGradientBoostingClassifier,
#     HistGradientBoostingRegressor,
#     RandomForestClassifier,
#     RandomForestRegressor,
# )
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.metrics import (
#     average_precision_score,
#     classification_report,
#     f1_score,
#     mean_absolute_error,
#     mean_squared_error,
#     precision_score,
#     recall_score,
#     r2_score,
#     roc_auc_score,
# )
# from sklearn.neural_network import MLPClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler

# from xgboost import XGBClassifier, XGBRegressor
# from lightgbm import LGBMClassifier, LGBMRegressor
# from catboost import CatBoostClassifier, CatBoostRegressor


# # ============================================================
# # CONFIGURAÇÃO
# # ============================================================

# BASE_DIR = Path(__file__).resolve().parents[1]

# DATASET_PATH = (
#     BASE_DIR
#     / "data"
#     / "irrigation_dataset_3.csv"
# )

# OUTPUT_DIR = (
#     BASE_DIR
#     / "models"
#     / "model_v9"
# )

# OUTPUT_DIR.mkdir(
#     parents=True,
#     exist_ok=True,
# )

# RANDOM_STATE = 42


# FEATURES = [
#     "soil_moisture_percent",
#     "temperature_c",
#     "air_humidity_percent",
#     "wind_speed_m_s",
#     "solar_radiation_kwh_m2",
#     "precipitation_mm",
#     "crop_stage",
#     "soil_moisture_lag_1h",
#     "soil_moisture_lag_3h",
#     "soil_moisture_lag_6h",
#     "kc",
#     "et0",
#     "etc",
#     "etc_6h",
#     "etc_24h",
#     "rain_6h",
#     "rain_24h",
#     "temperature_6h_mean",
#     "radiation_6h_sum",
#     "hour_sin",
#     "hour_cos",
#     "day_of_year_sin",
#     "day_of_year_cos",
# ]

# CLASSIFICATION_TARGET = "irrigation_event"
# REGRESSION_TARGET = "irrigation_depth_mm"


# # ============================================================
# # FUNÇÕES AUXILIARES
# # ============================================================

# def rmse(y_true, y_pred):
#     return np.sqrt(
#         mean_squared_error(
#             y_true,
#             y_pred,
#         )
#     )


# def find_best_threshold(
#     y_true,
#     probabilities,
# ):
#     """
#     Escolhe threshold usando SOMENTE a validação.

#     Critério:
#         maior F1.
#     """

#     thresholds = np.arange(
#         0.05,
#         0.96,
#         0.01,
#     )

#     best_threshold = 0.50
#     best_f1 = -1.0

#     for threshold in thresholds:

#         predictions = (
#             probabilities >= threshold
#         ).astype(int)

#         score = f1_score(
#             y_true,
#             predictions,
#             zero_division=0,
#         )

#         if score > best_f1:

#             best_f1 = score
#             best_threshold = threshold

#     return (
#         best_threshold,
#         best_f1,
#     )


# # ============================================================
# # CARREGAMENTO
# # ============================================================

# print("=" * 80)
# print("TREINAMENTO E BENCHMARK DO MODELO V9")
# print("=" * 80)

# print("\nCarregando dataset...")

# df = pd.read_csv(
#     DATASET_PATH,
#     parse_dates=["timestamp"],
# )

# df = (
#     df
#     .sort_values("timestamp")
#     .reset_index(drop=True)
# )

# print(
#     f"Registros: {len(df):,}"
# )

# print(
#     f"Colunas:   {len(df.columns)}"
# )

# print(
#     f"Período:   "
#     f"{df['timestamp'].min()} → "
#     f"{df['timestamp'].max()}"
# )


# # ============================================================
# # VERIFICAÇÕES
# # ============================================================

# required_columns = (
#     FEATURES
#     + [
#         CLASSIFICATION_TARGET,
#         REGRESSION_TARGET,
#         "timestamp",
#     ]
# )

# missing = [
#     column
#     for column in required_columns
#     if column not in df.columns
# ]

# if missing:
#     raise ValueError(
#         "Colunas ausentes:\n"
#         + "\n".join(missing)
#     )

# if df[FEATURES].isna().any().any():
#     raise ValueError(
#         "Existem NaN nas features."
#     )

# if df[
#     CLASSIFICATION_TARGET
# ].isna().any():
#     raise ValueError(
#         "Existem NaN no target de classificação."
#     )

# if df[
#     REGRESSION_TARGET
# ].isna().any():
#     raise ValueError(
#         "Existem NaN no target de regressão."
#     )


# # ============================================================
# # SPLIT TEMPORAL
# # ============================================================

# train = df[
#     df["timestamp"].dt.year <= 2023
# ].copy()

# validation = df[
#     df["timestamp"].dt.year == 2024
# ].copy()

# test = df[
#     df["timestamp"].dt.year == 2025
# ].copy()


# print("\n" + "=" * 80)
# print("SPLIT TEMPORAL")
# print("=" * 80)

# print(
#     f"Treino:     {len(train):,}"
# )

# print(
#     f"Validação:  {len(validation):,}"
# )

# print(
#     f"Teste:      {len(test):,}"
# )


# # ============================================================
# # MATRIZES DE CLASSIFICAÇÃO
# # ============================================================

# X_train = train[FEATURES]
# X_validation = validation[FEATURES]
# X_test = test[FEATURES]

# y_train_cls = train[
#     CLASSIFICATION_TARGET
# ].astype(int)

# y_validation_cls = validation[
#     CLASSIFICATION_TARGET
# ].astype(int)

# y_test_cls = test[
#     CLASSIFICATION_TARGET
# ].astype(int)


# print("\nEventos:")

# print(
#     f"Treino:    {y_train_cls.sum():,} "
#     f"({y_train_cls.mean() * 100:.4f}%)"
# )

# print(
#     f"Validação: {y_validation_cls.sum():,} "
#     f"({y_validation_cls.mean() * 100:.4f}%)"
# )

# print(
#     f"Teste:     {y_test_cls.sum():,} "
#     f"({y_test_cls.mean() * 100:.4f}%)"
# )


# # ============================================================
# # MATRIZES DE REGRESSÃO
# # ============================================================
# #
# # A regressão é treinada SOMENTE nos casos em que houve
# # irrigação real.
# #
# # Isso evita ensinar o regressor a prever zero.
# # ============================================================

# train_reg = train[
#     train[CLASSIFICATION_TARGET] == 1
# ].copy()

# validation_reg = validation[
#     validation[CLASSIFICATION_TARGET] == 1
# ].copy()

# test_reg = test[
#     test[CLASSIFICATION_TARGET] == 1
# ].copy()


# X_train_reg = train_reg[FEATURES]
# y_train_reg = train_reg[
#     REGRESSION_TARGET
# ]

# X_validation_reg = validation_reg[
#     FEATURES
# ]

# y_validation_reg = validation_reg[
#     REGRESSION_TARGET
# ]

# X_test_reg = test_reg[
#     FEATURES
# ]

# y_test_reg = test_reg[
#     REGRESSION_TARGET
# ]


# print("\nRegressão:")

# print(
#     f"Treino:    {len(train_reg):,} eventos"
# )

# print(
#     f"Validação: {len(validation_reg):,} eventos"
# )

# print(
#     f"Teste:     {len(test_reg):,} eventos"
# )


# # ============================================================
# # DEFINIÇÃO DOS MODELOS DE CLASSIFICAÇÃO
# # ============================================================

# classifier_candidates = []


# # ------------------------------------------------------------
# # 1. LOGISTIC REGRESSION
# # ------------------------------------------------------------

# classifier_candidates.append(
#     (
#         "logistic_regression",
#         Pipeline(
#             [
#                 (
#                     "scaler",
#                     StandardScaler(),
#                 ),
#                 (
#                     "model",
#                     LogisticRegression(
#                         max_iter=3000,
#                         class_weight="balanced",
#                         random_state=RANDOM_STATE,
#                     ),
#                 ),
#             ]
#         ),
#     )
# )


# # ------------------------------------------------------------
# # 2. RANDOM FOREST
# # ------------------------------------------------------------

# for n_estimators, max_depth in product(
#     [300],
#     [None, 15],
# ):

#     classifier_candidates.append(
#         (
#             f"random_forest_depth_{max_depth}",
#             RandomForestClassifier(
#                 n_estimators=n_estimators,
#                 max_depth=max_depth,
#                 class_weight="balanced",
#                 random_state=RANDOM_STATE,
#                 n_jobs=-1,
#             ),
#         )
#     )


# # ------------------------------------------------------------
# # 3. EXTRA TREES
# # ------------------------------------------------------------

# for n_estimators, max_depth in product(
#     [300],
#     [None, 15],
# ):

#     classifier_candidates.append(
#         (
#             f"extra_trees_depth_{max_depth}",
#             ExtraTreesClassifier(
#                 n_estimators=n_estimators,
#                 max_depth=max_depth,
#                 class_weight="balanced",
#                 random_state=RANDOM_STATE,
#                 n_jobs=-1,
#             ),
#         )
#     )


# # ------------------------------------------------------------
# # 4. HIST GRADIENT BOOSTING
# # ------------------------------------------------------------

# for learning_rate, max_leaf_nodes in product(
#     [0.05, 0.10],
#     [15, 31],
# ):

#     classifier_candidates.append(
#         (
#             (
#                 "hist_gradient_boosting_"
#                 f"lr_{learning_rate}_"
#                 f"leaf_{max_leaf_nodes}"
#             ),
#             HistGradientBoostingClassifier(
#                 learning_rate=learning_rate,
#                 max_iter=300,
#                 max_leaf_nodes=max_leaf_nodes,
#                 random_state=RANDOM_STATE,
#             ),
#         )
#     )


# # ------------------------------------------------------------
# # 5. XGBOOST
# # ------------------------------------------------------------

# for max_depth, learning_rate in product(
#     [4, 6],
#     [0.05, 0.10],
# ):

#     classifier_candidates.append(
#         (
#             (
#                 "xgboost_"
#                 f"depth_{max_depth}_"
#                 f"lr_{learning_rate}"
#             ),
#             XGBClassifier(
#                 n_estimators=400,
#                 max_depth=max_depth,
#                 learning_rate=learning_rate,
#                 subsample=0.9,
#                 colsample_bytree=0.9,
#                 objective="binary:logistic",
#                 eval_metric="logloss",
#                 random_state=RANDOM_STATE,
#                 n_jobs=-1,
#             ),
#         )
#     )


# # ------------------------------------------------------------
# # 6. LIGHTGBM
# # ------------------------------------------------------------

# for num_leaves, learning_rate in product(
#     [15, 31],
#     [0.05, 0.10],
# ):

#     classifier_candidates.append(
#         (
#             (
#                 "lightgbm_"
#                 f"leaves_{num_leaves}_"
#                 f"lr_{learning_rate}"
#             ),
#             LGBMClassifier(
#                 n_estimators=400,
#                 num_leaves=num_leaves,
#                 learning_rate=learning_rate,
#                 class_weight="balanced",
#                 random_state=RANDOM_STATE,
#                 n_jobs=-1,
#                 verbosity=-1,
#             ),
#         )
#     )


# # ------------------------------------------------------------
# # 7. CATBOOST
# # ------------------------------------------------------------

# for depth, learning_rate in product(
#     [5, 7],
#     [0.05, 0.10],
# ):

#     classifier_candidates.append(
#         (
#             (
#                 "catboost_"
#                 f"depth_{depth}_"
#                 f"lr_{learning_rate}"
#             ),
#             CatBoostClassifier(
#                 iterations=400,
#                 depth=depth,
#                 learning_rate=learning_rate,
#                 loss_function="Logloss",
#                 eval_metric="Logloss",
#                 verbose=False,
#                 random_seed=RANDOM_STATE,
#                 thread_count=-1,
#             ),
#         )
#     )


# # ------------------------------------------------------------
# # 8. MLP
# # ------------------------------------------------------------

# for hidden_layers in [
#     (64, 32),
#     (128, 64),
# ]:

#     classifier_candidates.append(
#         (
#             (
#                 "mlp_"
#                 + "_".join(
#                     map(
#                         str,
#                         hidden_layers,
#                     )
#                 )
#             ),
#             Pipeline(
#                 [
#                     (
#                         "scaler",
#                         StandardScaler(),
#                     ),
#                     (
#                         "model",
#                         MLPClassifier(
#                             hidden_layer_sizes=hidden_layers,
#                             activation="relu",
#                             solver="adam",
#                             alpha=0.0001,
#                             learning_rate_init=0.001,
#                             max_iter=300,
#                             early_stopping=True,
#                             validation_fraction=0.15,
#                             n_iter_no_change=15,
#                             random_state=RANDOM_STATE,
#                         ),
#                     ),
#                 ]
#             ),
#         )
#     )


# # ============================================================
# # BENCHMARK — CLASSIFICAÇÃO
# # ============================================================

# print("\n" + "=" * 80)
# print("BENCHMARK — CLASSIFICAÇÃO")
# print("=" * 80)

# classification_results = []

# trained_classifiers = {}


# for index, (
#     name,
#     model,
# ) in enumerate(
#     classifier_candidates,
#     start=1,
# ):

#     print(
#         f"\n[{index}/{len(classifier_candidates)}] "
#         f"Treinando {name}..."
#     )

#     model.fit(
#         X_train,
#         y_train_cls,
#     )

#     validation_probability = (
#         model.predict_proba(
#             X_validation
#         )[:, 1]
#     )

#     threshold, f1 = (
#         find_best_threshold(
#             y_validation_cls,
#             validation_probability,
#         )
#     )

#     validation_prediction = (
#         validation_probability
#         >= threshold
#     ).astype(int)

#     precision = precision_score(
#         y_validation_cls,
#         validation_prediction,
#         zero_division=0,
#     )

#     recall = recall_score(
#         y_validation_cls,
#         validation_prediction,
#         zero_division=0,
#     )

#     roc_auc = roc_auc_score(
#         y_validation_cls,
#         validation_probability,
#     )

#     pr_auc = average_precision_score(
#         y_validation_cls,
#         validation_probability,
#     )

#     classification_results.append(
#         {
#             "model": name,
#             "threshold": threshold,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#             "roc_auc": roc_auc,
#             "pr_auc": pr_auc,
#         }
#     )

#     trained_classifiers[name] = (
#         model,
#         threshold,
#     )

#     print(
#         f"  F1:      {f1:.4f}"
#     )

#     print(
#         f"  PR-AUC:  {pr_auc:.4f}"
#     )

#     print(
#         f"  ROC-AUC: {roc_auc:.4f}"
#     )


# classification_results_df = (
#     pd.DataFrame(
#         classification_results
#     )
#     .sort_values(
#         "f1",
#         ascending=False,
#     )
#     .reset_index(drop=True)
# )


# # ============================================================
# # RESULTADO CLASSIFICAÇÃO
# # ============================================================

# print("\n" + "=" * 80)
# print("RANKING — CLASSIFICAÇÃO")
# print("=" * 80)

# print(
#     classification_results_df.to_string(
#         index=False,
#     )
# )


# best_classifier_name = (
#     classification_results_df
#     .iloc[0]["model"]
# )

# best_classifier_threshold = (
#     classification_results_df
#     .iloc[0]["threshold"]
# )

# best_classifier = (
#     trained_classifiers[
#         best_classifier_name
#     ][0]
# )


# print("\nMelhor classificador:")
# print(
#     best_classifier_name
# )

# print(
#     f"Threshold: "
#     f"{best_classifier_threshold:.2f}"
# )


# # ============================================================
# # MODELOS DE REGRESSÃO
# # ============================================================

# regressor_candidates = []


# # ------------------------------------------------------------
# # 1. LINEAR REGRESSION
# # ------------------------------------------------------------

# regressor_candidates.append(
#     (
#         "linear_regression",
#         LinearRegression(),
#     )
# )


# # ------------------------------------------------------------
# # 2. RANDOM FOREST
# # ------------------------------------------------------------

# for n_estimators, max_depth in product(
#     [300],
#     [None, 15],
# ):

#     regressor_candidates.append(
#         (
#             f"random_forest_depth_{max_depth}",
#             RandomForestRegressor(
#                 n_estimators=n_estimators,
#                 max_depth=max_depth,
#                 random_state=RANDOM_STATE,
#                 n_jobs=-1,
#             ),
#         )
#     )


# # ------------------------------------------------------------
# # 3. EXTRA TREES
# # ------------------------------------------------------------

# for n_estimators, max_depth in product(
#     [300],
#     [None, 15],
# ):

#     regressor_candidates.append(
#         (
#             f"extra_trees_depth_{max_depth}",
#             ExtraTreesRegressor(
#                 n_estimators=n_estimators,
#                 max_depth=max_depth,
#                 random_state=RANDOM_STATE,
#                 n_jobs=-1,
#             ),
#         )
#     )


# # ------------------------------------------------------------
# # 4. HIST GRADIENT BOOSTING
# # ------------------------------------------------------------

# for learning_rate, max_leaf_nodes in product(
#     [0.05, 0.10],
#     [15, 31],
# ):

#     regressor_candidates.append(
#         (
#             (
#                 "hist_gradient_boosting_"
#                 f"lr_{learning_rate}_"
#                 f"leaf_{max_leaf_nodes}"
#             ),
#             HistGradientBoostingRegressor(
#                 learning_rate=learning_rate,
#                 max_iter=300,
#                 max_leaf_nodes=max_leaf_nodes,
#                 random_state=RANDOM_STATE,
#             ),
#         )
#     )


# # ------------------------------------------------------------
# # 5. XGBOOST
# # ------------------------------------------------------------

# for max_depth, learning_rate in product(
#     [4, 6],
#     [0.05, 0.10],
# ):

#     regressor_candidates.append(
#         (
#             (
#                 "xgboost_"
#                 f"depth_{max_depth}_"
#                 f"lr_{learning_rate}"
#             ),
#             XGBRegressor(
#                 n_estimators=400,
#                 max_depth=max_depth,
#                 learning_rate=learning_rate,
#                 subsample=0.9,
#                 colsample_bytree=0.9,
#                 objective="reg:squarederror",
#                 random_state=RANDOM_STATE,
#                 n_jobs=-1,
#             ),
#         )
#     )


# # ------------------------------------------------------------
# # 6. LIGHTGBM
# # ------------------------------------------------------------

# for num_leaves, learning_rate in product(
#     [15, 31],
#     [0.05, 0.10],
# ):

#     regressor_candidates.append(
#         (
#             (
#                 "lightgbm_"
#                 f"leaves_{num_leaves}_"
#                 f"lr_{learning_rate}"
#             ),
#             LGBMRegressor(
#                 n_estimators=400,
#                 num_leaves=num_leaves,
#                 learning_rate=learning_rate,
#                 random_state=RANDOM_STATE,
#                 n_jobs=-1,
#                 verbosity=-1,
#             ),
#         )
#     )


# # ------------------------------------------------------------
# # 7. CATBOOST
# # ------------------------------------------------------------

# for depth, learning_rate in product(
#     [5, 7],
#     [0.05, 0.10],
# ):

#     regressor_candidates.append(
#         (
#             (
#                 "catboost_"
#                 f"depth_{depth}_"
#                 f"lr_{learning_rate}"
#             ),
#             CatBoostRegressor(
#                 iterations=400,
#                 depth=depth,
#                 learning_rate=learning_rate,
#                 loss_function="RMSE",
#                 verbose=False,
#                 random_seed=RANDOM_STATE,
#                 thread_count=-1,
#             ),
#         )
#     )


# # ============================================================
# # BENCHMARK — REGRESSÃO
# # ============================================================

# print("\n" + "=" * 80)
# print("BENCHMARK — REGRESSÃO")
# print("=" * 80)

# regression_results = []

# trained_regressors = {}


# for index, (
#     name,
#     model,
# ) in enumerate(
#     regressor_candidates,
#     start=1,
# ):

#     print(
#         f"\n[{index}/{len(regressor_candidates)}] "
#         f"Treinando {name}..."
#     )

#     model.fit(
#         X_train_reg,
#         y_train_reg,
#     )

#     validation_prediction = (
#         model.predict(
#             X_validation_reg
#         )
#     )

#     mae = mean_absolute_error(
#         y_validation_reg,
#         validation_prediction,
#     )

#     rmse_value = rmse(
#         y_validation_reg,
#         validation_prediction,
#     )

#     r2 = r2_score(
#         y_validation_reg,
#         validation_prediction,
#     )

#     regression_results.append(
#         {
#             "model": name,
#             "mae": mae,
#             "rmse": rmse_value,
#             "r2": r2,
#         }
#     )

#     trained_regressors[name] = model

#     print(
#         f"  MAE:  {mae:.4f} mm"
#     )

#     print(
#         f"  RMSE: {rmse_value:.4f} mm"
#     )

#     print(
#         f"  R²:   {r2:.4f}"
#     )


# regression_results_df = (
#     pd.DataFrame(
#         regression_results
#     )
#     .sort_values(
#         "mae",
#         ascending=True,
#     )
#     .reset_index(drop=True)
# )


# # ============================================================
# # RESULTADO REGRESSÃO
# # ============================================================

# print("\n" + "=" * 80)
# print("RANKING — REGRESSÃO")
# print("=" * 80)

# print(
#     regression_results_df.to_string(
#         index=False,
#     )
# )


# best_regressor_name = (
#     regression_results_df
#     .iloc[0]["model"]
# )

# best_regressor = (
#     trained_regressors[
#         best_regressor_name
#     ]
# )


# print("\nMelhor regressor:")
# print(
#     best_regressor_name
# )


# # ============================================================
# # TESTE FINAL — CLASSIFICAÇÃO
# # ============================================================

# print("\n" + "=" * 80)
# print("TESTE FINAL — 2025")
# print("=" * 80)

# test_probability = (
#     best_classifier
#     .predict_proba(
#         X_test
#     )[:, 1]
# )

# test_prediction = (
#     test_probability
#     >= best_classifier_threshold
# ).astype(int)


# test_precision = precision_score(
#     y_test_cls,
#     test_prediction,
#     zero_division=0,
# )

# test_recall = recall_score(
#     y_test_cls,
#     test_prediction,
#     zero_division=0,
# )

# test_f1 = f1_score(
#     y_test_cls,
#     test_prediction,
#     zero_division=0,
# )

# test_roc_auc = roc_auc_score(
#     y_test_cls,
#     test_probability,
# )

# test_pr_auc = average_precision_score(
#     y_test_cls,
#     test_probability,
# )


# print("\nCLASSIFICAÇÃO")

# print(
#     f"Modelo:    "
#     f"{best_classifier_name}"
# )

# print(
#     f"Threshold: "
#     f"{best_classifier_threshold:.2f}"
# )

# print(
#     f"Precision: "
#     f"{test_precision:.4f}"
# )

# print(
#     f"Recall:    "
#     f"{test_recall:.4f}"
# )

# print(
#     f"F1:        "
#     f"{test_f1:.4f}"
# )

# print(
#     f"ROC-AUC:   "
#     f"{test_roc_auc:.4f}"
# )

# print(
#     f"PR-AUC:    "
#     f"{test_pr_auc:.4f}"
# )

# print(
#     "\nClassification report:"
# )

# print(
#     classification_report(
#         y_test_cls,
#         test_prediction,
#         digits=4,
#         zero_division=0,
#     )
# )


# # ============================================================
# # TESTE FINAL — REGRESSÃO
# # ============================================================

# test_reg_prediction = (
#     best_regressor.predict(
#         X_test_reg
#     )
# )

# test_mae = mean_absolute_error(
#     y_test_reg,
#     test_reg_prediction,
# )

# test_rmse = rmse(
#     y_test_reg,
#     test_reg_prediction,
# )

# test_r2 = r2_score(
#     y_test_reg,
#     test_reg_prediction,
# )


# print("\nREGRESSÃO")

# print(
#     f"Modelo: "
#     f"{best_regressor_name}"
# )

# print(
#     f"MAE:  "
#     f"{test_mae:.4f} mm"
# )

# print(
#     f"RMSE: "
#     f"{test_rmse:.4f} mm"
# )

# print(
#     f"R²:   "
#     f"{test_r2:.4f}"
# )


# # ============================================================
# # END-TO-END
# # ============================================================

# print("\n" + "=" * 80)
# print("AVALIAÇÃO END-TO-END")
# print("=" * 80)

# end_to_end_prediction = np.zeros(
#     len(test)
# )

# predicted_irrigation_mask = (
#     test_prediction == 1
# )

# if predicted_irrigation_mask.any():

#     positive_features = X_test[
#         predicted_irrigation_mask
#     ]

#     end_to_end_prediction[
#         predicted_irrigation_mask
#     ] = best_regressor.predict(
#         positive_features
#     )

# end_to_end_real = test[
#     REGRESSION_TARGET
# ].to_numpy()


# end_to_end_mae = mean_absolute_error(
#     end_to_end_real,
#     end_to_end_prediction,
# )

# end_to_end_rmse = rmse(
#     end_to_end_real,
#     end_to_end_prediction,
# )

# end_to_end_r2 = r2_score(
#     end_to_end_real,
#     end_to_end_prediction,
# )


# print(
#     f"MAE:  "
#     f"{end_to_end_mae:.4f} mm"
# )

# print(
#     f"RMSE: "
#     f"{end_to_end_rmse:.4f} mm"
# )

# print(
#     f"R²:   "
#     f"{end_to_end_r2:.4f}"
# )


# # ============================================================
# # IMPORTÂNCIA DAS FEATURES
# # ============================================================

# print("\n" + "=" * 80)
# print("IMPORTÂNCIA DAS FEATURES")
# print("=" * 80)


# def save_feature_importance(
#     model,
#     model_name,
# ):

#     estimator = model

#     if isinstance(
#         model,
#         Pipeline,
#     ):
#         estimator = model.named_steps[
#             "model"
#         ]

#     if not hasattr(
#         estimator,
#         "feature_importances_",
#     ):
#         return

#     importance = pd.DataFrame(
#         {
#             "feature": FEATURES,
#             "importance": (
#                 estimator
#                 .feature_importances_
#             ),
#         }
#     ).sort_values(
#         "importance",
#         ascending=False,
#     )

#     path = (
#         OUTPUT_DIR
#         / f"feature_importance_{model_name}.csv"
#     )

#     importance.to_csv(
#         path,
#         index=False,
#     )

#     print(
#         f"\nTop features — {model_name}:"
#     )

#     print(
#         importance.head(15)
#         .to_string(index=False)
#     )


# save_feature_importance(
#     best_classifier,
#     f"classifier_{best_classifier_name}",
# )

# save_feature_importance(
#     best_regressor,
#     f"regressor_{best_regressor_name}",
# )


# # ============================================================
# # SALVAR MODELOS
# # ============================================================

# classifier_path = (
#     OUTPUT_DIR
#     / f"classifier_{best_classifier_name}.joblib"
# )

# regressor_path = (
#     OUTPUT_DIR
#     / f"regressor_{best_regressor_name}.joblib"
# )


# joblib.dump(
#     best_classifier,
#     classifier_path,
# )

# joblib.dump(
#     best_regressor,
#     regressor_path,
# )


# # ============================================================
# # SALVAR COMPARAÇÕES
# # ============================================================

# classification_results_df.to_csv(
#     OUTPUT_DIR
#     / "classification_comparison.csv",
#     index=False,
# )

# regression_results_df.to_csv(
#     OUTPUT_DIR
#     / "regression_comparison.csv",
#     index=False,
# )


# # ============================================================
# # RESUMO
# # ============================================================

# summary = pd.DataFrame(
#     [
#         {
#             "classifier": (
#                 best_classifier_name
#             ),
#             "classifier_threshold": (
#                 best_classifier_threshold
#             ),
#             "test_precision": (
#                 test_precision
#             ),
#             "test_recall": (
#                 test_recall
#             ),
#             "test_f1": (
#                 test_f1
#             ),
#             "test_roc_auc": (
#                 test_roc_auc
#             ),
#             "test_pr_auc": (
#                 test_pr_auc
#             ),
#             "regressor": (
#                 best_regressor_name
#             ),
#             "test_mae_mm": (
#                 test_mae
#             ),
#             "test_rmse_mm": (
#                 test_rmse
#             ),
#             "test_r2": (
#                 test_r2
#             ),
#             "end_to_end_mae_mm": (
#                 end_to_end_mae
#             ),
#             "end_to_end_rmse_mm": (
#                 end_to_end_rmse
#             ),
#             "end_to_end_r2": (
#                 end_to_end_r2
#             ),
#         }
#     ]
# )


# summary.to_csv(
#     OUTPUT_DIR
#     / "v9_summary.csv",
#     index=False,
# )


# # ============================================================
# # FINAL
# # ============================================================

# print("\n" + "=" * 80)
# print("V9 FINALIZADO")
# print("=" * 80)

# print(
#     "\nMelhor classificador:"
# )

# print(
#     best_classifier_name
# )

# print(
#     f"F1 teste: {test_f1:.4f}"
# )

# print(
#     "\nMelhor regressor:"
# )

# print(
#     best_regressor_name
# )

# print(
#     f"MAE teste: {test_mae:.4f} mm"
# )

# print(
#     "\nModelos salvos em:"
# )

# print(
#     OUTPUT_DIR
# )

# print(
#     "\nArquivos:"
# )

# print(
#     classifier_path.name
# )

# print(
#     regressor_path.name
# )

# print(
#     "\nComparações:"
# )

# print(
#     "classification_comparison.csv"
# )

# print(
#     "regression_comparison.csv"
# )

# print(
#     "v9_summary.csv"
# )

# print("\n" + "=" * 80)



from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    BASE_DIR
    / "ml"
    / "data"
    / "irrigation_dataset_3.csv"
)

MODELS_DIR = (
    BASE_DIR
    / "ml"
    / "models"
    / "model_v9"
)

CLASSIFIER_PATH = (
    MODELS_DIR
    / "classifier_random_forest_depth_None.joblib"
)

REGRESSOR_PATH = (
    MODELS_DIR
    / "regressor_catboost_depth_7_lr_0.1.joblib"
)

THRESHOLD = 0.25


# ============================================================
# FEATURES UTILIZADAS PELOS MODELOS
# ============================================================

FEATURES = [
    "soil_moisture_percent",
    "temperature_c",
    "air_humidity_percent",
    "wind_speed_m_s",
    "solar_radiation_kwh_m2",
    "precipitation_mm",
    "crop_stage",
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


# ============================================================
# CARREGAMENTO
# ============================================================

print("=" * 80)
print("TESTE DE INFERÊNCIA — V9 FINAL")
print("=" * 80)

print("\nCarregando modelos...")

classifier = joblib.load(CLASSIFIER_PATH)
regressor = joblib.load(REGRESSOR_PATH)

print(f"Classificador: {CLASSIFIER_PATH.name}")
print(f"Regressor:     {REGRESSOR_PATH.name}")
print(f"Threshold:     {THRESHOLD}")


print("\nCarregando dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Registros carregados: {len(df):,}")
print(f"Colunas: {len(df.columns)}")


# ============================================================
# FUNÇÃO DE INFERÊNCIA
# ============================================================

def predict_row(row):
    """
    Executa a inferência completa para uma linha do dataset.
    """

    X = pd.DataFrame(
        [[row[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    probability = float(
        classifier.predict_proba(X)[0, 1]
    )

    irrigation_required = probability >= THRESHOLD

    if irrigation_required:
        irrigation_depth = float(
            regressor.predict(X)[0]
        )
        irrigation_depth = max(0.0, irrigation_depth)
    else:
        irrigation_depth = 0.0

    return probability, irrigation_required, irrigation_depth


# ============================================================
# TESTE 1 — VARIAÇÃO CONTROLADA DA UMIDADE
# ============================================================

print("\n")
print("=" * 80)
print("TESTE 1 — VARIAÇÃO CONTROLADA DE SOIL_MOISTURE_PERCENT")
print("=" * 80)

# Pegamos uma linha real do dataset como base.
# Assim, todas as outras variáveis pertencem a uma
# combinação realmente existente no dataset.

base_row = df.iloc[
    df["irrigation_event"].astype(int).eq(1).idxmax()
].copy()

base_values = {
    feature: base_row[feature]
    for feature in FEATURES
}

moisture_values = [
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    100.0,
]

results_controlled = []

print("\nLinha-base utilizada:")
print(
    f"  crop_stage = {base_row['crop_stage']}"
)
print(
    f"  kc         = {base_row['kc']:.3f}"
)
print(
    f"  etc        = {base_row['etc']:.3f}"
)
print(
    f"  et0        = {base_row['et0']:.3f}"
)

print("\n")
print(
    f"{'Umidade':>10} "
    f"{'Probabilidade':>16} "
    f"{'Irrigar':>10} "
    f"{'Lâmina (mm)':>14}"
)
print("-" * 56)

for moisture in moisture_values:

    test_row = base_values.copy()

    test_row["soil_moisture_percent"] = moisture

    probability, irrigation_required, irrigation_depth = (
        predict_row(test_row)
    )

    results_controlled.append(
        {
            "soil_moisture_percent": moisture,
            "probability": probability,
            "irrigation_required": irrigation_required,
            "irrigation_depth_mm": irrigation_depth,
        }
    )

    print(
        f"{moisture:>9.1f}% "
        f"{probability:>15.6f} "
        f"{str(irrigation_required):>10} "
        f"{irrigation_depth:>13.3f}"
    )


controlled_df = pd.DataFrame(results_controlled)


# ============================================================
# TESTE 2 — REGISTROS REAIS COM IRRIGAÇÃO
# ============================================================

print("\n")
print("=" * 80)
print("TESTE 2 — REGISTROS REAIS COM IRRIGAÇÃO")
print("=" * 80)

event_rows = df[
    df["irrigation_event"].astype(int) == 1
].copy()

# Seleciona pontos aproximadamente distribuídos
# ao longo da faixa de umidade.
event_rows = (
    event_rows
    .sort_values("soil_moisture_percent")
)

indices = np.linspace(
    0,
    len(event_rows) - 1,
    num=min(10, len(event_rows)),
    dtype=int,
)

selected_events = event_rows.iloc[indices]

event_results = []

print(
    f"\n{'Umidade':>9} "
    f"{'Estágio':>8} "
    f"{'Real':>10} "
    f"{'Prob.':>10} "
    f"{'Pred.':>10} "
    f"{'Erro':>10} "
    f"{'Classe':>10}"
)

print("-" * 75)

for _, row in selected_events.iterrows():

    probability, irrigation_required, predicted_depth = (
        predict_row(row)
    )

    real_depth = float(
        row["irrigation_depth_mm"]
    )

    error = predicted_depth - real_depth

    classification_hit = (
        irrigation_required
        == bool(row["irrigation_event"])
    )

    event_results.append(
        {
            "soil_moisture_percent":
                row["soil_moisture_percent"],

            "crop_stage":
                row["crop_stage"],

            "real_depth_mm":
                real_depth,

            "probability":
                probability,

            "predicted_depth_mm":
                predicted_depth,

            "error_mm":
                error,

            "classification_hit":
                classification_hit,
        }
    )

    print(
        f"{row['soil_moisture_percent']:>8.2f}% "
        f"{row['crop_stage']:>8.0f} "
        f"{real_depth:>9.3f} "
        f"{probability:>9.6f} "
        f"{predicted_depth:>9.3f} "
        f"{error:>+9.3f} "
        f"{str(classification_hit):>10}"
    )


event_results_df = pd.DataFrame(event_results)


# ============================================================
# TESTE 3 — REGISTROS REAIS SEM IRRIGAÇÃO
# ============================================================

print("\n")
print("=" * 80)
print("TESTE 3 — REGISTROS REAIS SEM IRRIGAÇÃO")
print("=" * 80)

no_event_rows = df[
    df["irrigation_event"].astype(int) == 0
].copy()

# Seleciona pontos distribuídos pela faixa de armazenamento.
no_event_rows = (
    no_event_rows
    .sort_values("soil_moisture_percent")
)

indices = np.linspace(
    0,
    len(no_event_rows) - 1,
    num=min(10, len(no_event_rows)),
    dtype=int,
)

selected_no_events = no_event_rows.iloc[indices]

no_event_results = []

print(
    f"\n{'Umidade':>9} "
    f"{'Estágio':>8} "
    f"{'Prob.':>10} "
    f"{'Irrigar':>10} "
    f"{'Lâmina':>12} "
    f"{'Correto':>10}"
)

print("-" * 70)

for _, row in selected_no_events.iterrows():

    probability, irrigation_required, predicted_depth = (
        predict_row(row)
    )

    classification_hit = (
        irrigation_required
        == bool(row["irrigation_event"])
    )

    no_event_results.append(
        {
            "soil_moisture_percent":
                row["soil_moisture_percent"],

            "crop_stage":
                row["crop_stage"],

            "probability":
                probability,

            "irrigation_required":
                irrigation_required,

            "predicted_depth_mm":
                predicted_depth,

            "classification_hit":
                classification_hit,
        }
    )

    print(
        f"{row['soil_moisture_percent']:>8.2f}% "
        f"{row['crop_stage']:>8.0f} "
        f"{probability:>9.6f} "
        f"{str(irrigation_required):>10} "
        f"{predicted_depth:>11.3f} "
        f"{str(classification_hit):>10}"
    )


no_event_results_df = pd.DataFrame(
    no_event_results
)


# ============================================================
# TESTE 4 — EXEMPLOS POR ESTÁGIO
# ============================================================

print("\n")
print("=" * 80)
print("TESTE 4 — INFERÊNCIA POR ESTÁGIO FENOLÓGICO")
print("=" * 80)

stage_results = []

for stage in sorted(
    df["crop_stage"].dropna().unique()
):

    stage_df = df[
        df["crop_stage"] == stage
    ].copy()

    # Preferimos uma linha que realmente seja
    # evento de irrigação para avaliar a lâmina.
    stage_events = stage_df[
        stage_df["irrigation_event"].astype(int) == 1
    ]

    if len(stage_events) > 0:
        row = stage_events.iloc[
            len(stage_events) // 2
        ]
    else:
        row = stage_df.iloc[
            len(stage_df) // 2
        ]

    probability, irrigation_required, predicted_depth = (
        predict_row(row)
    )

    real_event = bool(
        row["irrigation_event"]
    )

    real_depth = float(
        row["irrigation_depth_mm"]
    )

    stage_results.append(
        {
            "crop_stage": stage,
            "kc": row["kc"],
            "etc": row["etc"],
            "taw_mm": row["taw_mm"],
            "raw_mm": row["raw_mm"],
            "soil_moisture_percent":
                row["soil_moisture_percent"],
            "real_event": real_event,
            "predicted_event":
                irrigation_required,
            "probability":
                probability,
            "real_depth_mm":
                real_depth,
            "predicted_depth_mm":
                predicted_depth,
            "error_mm":
                predicted_depth - real_depth,
        }
    )


stage_results_df = pd.DataFrame(stage_results)

print(
    f"\n{'Estágio':>8} "
    f"{'Kc':>7} "
    f"{'ETc':>8} "
    f"{'TAW':>8} "
    f"{'RAW':>8} "
    f"{'Umidade':>10} "
    f"{'Prob.':>10} "
    f"{'Real':>9} "
    f"{'Pred.':>9} "
    f"{'Erro':>9}"
)

print("-" * 105)

for _, row in stage_results_df.iterrows():

    print(
        f"{row['crop_stage']:>8.0f} "
        f"{row['kc']:>7.3f} "
        f"{row['etc']:>7.3f} "
        f"{row['taw_mm']:>7.3f} "
        f"{row['raw_mm']:>7.3f} "
        f"{row['soil_moisture_percent']:>9.2f}% "
        f"{row['probability']:>9.6f} "
        f"{row['real_depth_mm']:>8.3f} "
        f"{row['predicted_depth_mm']:>8.3f} "
        f"{row['error_mm']:>+8.3f}"
    )


# ============================================================
# RESUMO DOS TESTES
# ============================================================

print("\n")
print("=" * 80)
print("RESUMO DOS TESTES DE INFERÊNCIA")
print("=" * 80)

print("\n1. Teste controlado:")
print(
    "   Foram avaliados valores de armazenamento relativo "
    "entre 20% e 100%."
)

print("\n2. Eventos reais:")
if len(event_results_df) > 0:

    event_mae = (
        event_results_df["error_mm"]
        .abs()
        .mean()
    )

    event_accuracy = (
        event_results_df["classification_hit"]
        .mean()
    )

    print(
        f"   MAE das lâminas: {event_mae:.4f} mm"
    )

    print(
        f"   Acurácia da classificação: "
        f"{event_accuracy:.4f}"
    )

print("\n3. Não eventos reais:")
if len(no_event_results_df) > 0:

    no_event_accuracy = (
        no_event_results_df["classification_hit"]
        .mean()
    )

    print(
        f"   Acurácia da classificação: "
        f"{no_event_accuracy:.4f}"
    )

print("\n4. Estágios fenológicos:")
print(
    "   Foram avaliados exemplos dos quatro estágios "
    "do ciclo do coentro."
)

print("\n")
print("=" * 80)
print("TESTES FINALIZADOS")
print("=" * 80)