#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protocole Scientifique de Recherche Multi-Sites PM10 (Cadre Lamto)
Analyse Comparative, Transférabilité Spatiale et Dynamique EXT -> INT

Axe 1 : Modèles Dédiés Locaux (10 Sites x 2 Capteurs EXT/INT)
Axe 2 : Transférabilité Spatiale Inter-Sites (Sites 1..7 -> Sites 8..10)
Axe 3 : Modélisation Physique du Transfert d'Infiltration (EXT -> INT)

Découpage temporel strict : 80% Train / 10% Validation / 10% Test
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & DOSSIERS DE SORTIE
# -----------------------------------------------------------------------------
DATA_DIR = "./processed_data/by_site"
MASTER_FILE = "./processed_data/dataset_master_daily_all_sites.csv"
RESULTS_DIR = "./results"
MODELS_DIR = "./models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. FONCTIONS DE TRAITEMENT ET FEATURE ENGINEERING (CADRE LAMTO)
# -----------------------------------------------------------------------------
def get_climate_season(month):
    """4 régimes climatiques de savane (Table 5 - N'Datchoh et al. 2025)."""
    if month in [11, 12, 1, 2]:
        return "Grande Saison Sèche"
    elif month in [3, 4, 5, 6, 7]:
        return "Grande Saison des Pluies"
    elif month == 8:
        return "Petite Saison Sèche"
    else:
        return "Petite Saison des Pluies"

def weed_imputation(df, target_col='PM10'):
    """Imputation saisonnière par médiane du jour calendaire (Weed et al. 2022)."""
    df = df.copy()
    if df[target_col].isna().sum() == 0:
        return df
    df['day_of_year'] = df.index.dayofyear
    medians = df.groupby('day_of_year')[target_col].transform('median')
    df[target_col] = df[target_col].fillna(medians).fillna(df[target_col].median()).ffill().bfill()
    df.drop(columns=['day_of_year'], inplace=True)
    return df

def build_features(df_input, target_col='PM10'):
    """Feature engineering autorégressif et thermodynamique."""
    df = df_input.copy().sort_index()
    df = weed_imputation(df, target_col)

    # Variables temporelles
    df['Month'] = df.index.month
    df['DayOfWeek'] = df.index.dayofweek
    df['Season'] = df['Month'].apply(get_climate_season)

    # Variables autorégressives (Table 4 de l'article)
    df['PM10_lag1'] = df[target_col].shift(1)
    df['PM10_lag2'] = df[target_col].shift(2)
    df['PM10_lag3'] = df[target_col].shift(3)

    # Variables thermodynamiques
    if 'temp_c' in df.columns:
        df['temp_c_lag1'] = df['temp_c'].shift(1)
    if 'dewpoint_c' in df.columns:
        df['dewpoint_c_lag1'] = df['dewpoint_c'].shift(1)
    if 'temp_c' in df.columns and 'dewpoint_c' in df.columns:
        df['dewpoint_deficit'] = df['temp_c'] - df['dewpoint_c']
    if 'humidity' in df.columns:
        df['humidity_lag1'] = df['humidity'].shift(1)
    if 'pressure_hpa' in df.columns:
        df['pressure_hpa_lag1'] = df['pressure_hpa'].shift(1)

    df.dropna(subset=['PM10_lag1', 'PM10_lag2', 'PM10_lag3'], inplace=True)
    return df

def compute_metrics(y_true, y_pred):
    """Calcul standard RMSE, MAE, R2."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

# -----------------------------------------------------------------------------
# 3. ENTRAÎNEMENT DES 4 ARCHITECTURES DE LAMTO
# -----------------------------------------------------------------------------
def train_4_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """Entraîne SARIMAX, Random Forest, XGBoost et LightGBM."""
    results = {}
    models = {}

    # 1. Random Forest (Champion Lamto : n=100, max_depth=10)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    models['Random Forest'] = rf
    results['Random Forest'] = (compute_metrics(y_test, pred_rf), pred_rf)

    # 2. XGBoost (n=100, lr=0.1, max_depth=3)
    xgb_mod = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, verbosity=0)
    xgb_mod.fit(X_train, y_train)
    pred_xgb = xgb_mod.predict(X_test)
    models['XGBoost'] = xgb_mod
    results['XGBoost'] = (compute_metrics(y_test, pred_xgb), pred_xgb)

    # 3. LightGBM (gbdt, n=100, metric=rmse)
    lgb_mod = lgb.LGBMRegressor(boosting_type='gbdt', n_estimators=100, objective='regression', random_state=42, verbose=-1)
    lgb_mod.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10, verbose=False)])
    pred_lgb = lgb_mod.predict(X_test)
    models['LightGBM'] = lgb_mod
    results['LightGBM'] = (compute_metrics(y_test, pred_lgb), pred_lgb)

    # 4. SARIMAX (1,0,2) x (2,0,2,12)
    try:
        sarimax = SARIMAX(y_train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False)
        sarimax_fit = sarimax.fit(disp=False, maxiter=30)
        pred_sar = sarimax_fit.forecast(steps=len(y_test))
        models['SARIMAX'] = sarimax_fit
        results['SARIMAX'] = (compute_metrics(y_test, pred_sar), pred_sar.values)
    except Exception:
        # En cas d'erreur de convergence
        pred_sar = np.full(len(y_test), y_train.mean())
        results['SARIMAX'] = (compute_metrics(y_test, pred_sar), pred_sar)

    return models, results

# -----------------------------------------------------------------------------
# 4. AXE 1 : MODÈLES DÉDIÉS LOCAUX (SITE PAR SITE & CAPTEUR PAR CAPTEUR)
# -----------------------------------------------------------------------------
def run_axe1_local_benchmarks():
    print("\n" + "=" * 75)
    print("🔹 AXE 1 : MODÈLES DÉDIÉS LOCAUX (Intra-Site & Intra-Point)")
    print("=" * 75)

    daily_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_daily.csv")))
    if not daily_files:
        print("[!] Aucun fichier _daily.csv trouvé dans", DATA_DIR)
        return pd.DataFrame(), {}

    records = []
    trained_local_artifacts = {}

    for file_path in daily_files:
        filename = os.path.basename(file_path)
        # Extraction du nom du site et du type (EXT ou INT)
        match = re.match(r"(SITE_\d+)_(P1_EXT|P2_INT)_daily\.csv", filename)
        if not match:
            continue
        site_name, location_type = match.group(1).replace("_", " "), match.group(2).replace("_", " ")

        df = pd.read_csv(file_path)
        date_col = 'date' if 'date' in df.columns else df.columns[0]
        df['date'] = pd.to_datetime(df[date_col])
        df.set_index('date', inplace=True)

        if 'PM10' not in df.columns or len(df.dropna(subset=['PM10'])) < 20:
            continue

        df_feat = build_features(df, 'PM10')
        feature_cols = [c for c in ['PM10_lag1', 'PM10_lag2', 'PM10_lag3', 'Month', 'DayOfWeek', 'temp_c', 'temp_c_lag1', 'dewpoint_c', 'dewpoint_deficit', 'humidity', 'humidity_lag1', 'pressure_hpa'] if c in df_feat.columns]

        # Découpage strict 80% / 10% / 10%
        n = len(df_feat)
        n_train, n_val = int(n * 0.80), int(n * 0.90)

        X_train, y_train = df_feat.iloc[:n_train][feature_cols], df_feat.iloc[:n_train]['PM10']
        X_val, y_val = df_feat.iloc[n_train:n_val][feature_cols], df_feat.iloc[n_train:n_val]['PM10']
        X_test, y_test = df_feat.iloc[n_val:][feature_cols], df_feat.iloc[n_val:]['PM10']

        if len(X_test) < 3:
            continue

        models, results = train_4_models(X_train, y_train, X_val, y_val, X_test, y_test)

        # Recherche du champion sur ce site
        best_model_name = max(results.keys(), key=lambda m: results[m][0]['R2'])
        best_r2 = results[best_model_name][0]['R2']
        best_rmse = results[best_model_name][0]['RMSE']

        trained_local_artifacts[f"{site_name}_{location_type}"] = {
            'site': site_name,
            'location': location_type,
            'best_model_name': best_model_name,
            'model': models[best_model_name],
            'feature_cols': feature_cols,
            'X_test': X_test,
            'y_test': y_test,
            'preds': results[best_model_name][1]
        }

        for model_name, (metrics, _) in results.items():
            records.append({
                'Site': site_name,
                'Capteur': location_type,
                'Modèle': model_name,
                'R²': round(metrics['R2'], 3),
                'RMSE (µg/m³)': round(metrics['RMSE'], 2),
                'MAE (µg/m³)': round(metrics['MAE'], 2),
                'N_Train': len(X_train),
                'N_Test': len(X_test),
                'Est_Champion': "⭐" if model_name == best_model_name else ""
            })

        print(f"  • {site_name} ({location_type}) ➔ Champion : {best_model_name} (R² = {best_r2:.3f}, RMSE = {best_rmse:.2f} µg/m³)")

    df_results_axe1 = pd.DataFrame(records)
    out_file = os.path.join(RESULTS_DIR, "leaderboard_axe1_local_models.csv")
    df_results_axe1.to_csv(out_file, index=False)
    print(f"\n📊 Résultats Axe 1 exportés : {out_file}")
    return df_results_axe1, trained_local_artifacts

# -----------------------------------------------------------------------------
# 5. AXE 2 : TRANSFÉRABILITÉ SPATIALE INTER-SITES (CROSS-SITE GENERALIZATION)
# -----------------------------------------------------------------------------
def run_axe2_spatial_transfer():
    print("\n" + "=" * 75)
    print("🔹 AXE 2 : TRANSFÉRABILITÉ SPATIALE INTER-SITES")
    print("  (Entraînement sur SITES 1 à 7 ➔ Test de Généralisation sur SITES 8 à 10)")
    print("=" * 75)

    if not os.path.exists(MASTER_FILE):
        return pd.DataFrame()

    df_master = pd.read_csv(MASTER_FILE)
    df_master['date'] = pd.to_datetime(df_master['date'])
    df_master.set_index('date', inplace=True)

    transfer_results = []

    for loc in ['EXT', 'INT']:
        loc_label = "P1 EXT (Ambiant)" if loc == 'EXT' else "P2 INT (Intérieur)"
        print(f"\n  [+] Évaluation du Transfert Spatial pour : {loc_label}")

        df_loc = df_master[df_master['location'] == loc].copy()
        sites = sorted(df_loc['site'].unique())

        # Sites d'entraînement (ex: Sites 1 à 7) vs Sites de Test non vus (Sites 8, 9, 10)
        train_sites = [s for s in sites if any(f"SITE {i}" in s for i in range(1, 8))]
        test_sites = [s for s in sites if any(f"SITE {i}" in s for i in range(8, 11))]

        if not train_sites or not test_sites:
            train_sites, test_sites = sites[:-2], sites[-2:]

        # Préparation du jeu d'entraînement régional
        train_dfs = []
        for s in train_sites:
            sub = df_loc[df_loc['site'] == s].copy()
            if len(sub) > 20:
                train_dfs.append(build_features(sub, 'PM10'))

        if not train_dfs:
            continue

        df_train_all = pd.concat(train_dfs).sort_index()
        feature_cols = [c for c in ['PM10_lag1', 'PM10_lag2', 'PM10_lag3', 'Month', 'DayOfWeek', 'temp_c', 'dewpoint_deficit', 'humidity'] if c in df_train_all.columns]

        X_train_reg = df_train_all[feature_cols]
        y_train_reg = df_train_all['PM10']

        # Entraînement du modèle régional (Random Forest)
        rf_regional = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_regional.fit(X_train_reg, y_train_reg)

        # Test sur chaque site non vu
        for s_test in test_sites:
            sub_test = df_loc[df_loc['site'] == s_test].copy()
            if len(sub_test) < 10:
                continue
            df_feat_test = build_features(sub_test, 'PM10')
            X_test_unseen = df_feat_test[feature_cols]
            y_test_unseen = df_feat_test['PM10']

            preds_unseen = rf_regional.predict(X_test_unseen)
            m = compute_metrics(y_test_unseen, preds_unseen)

            transfer_results.append({
                'Type_Capteur': loc,
                'Site_Test_Inédit': s_test,
                'Modèle': 'Random Forest Régional (Transféré)',
                'R²': round(m['R2'], 3),
                'RMSE (µg/m³)': round(m['RMSE'], 2),
                'MAE (µg/m³)': round(m['MAE'], 2),
                'N_Obs_Test': len(y_test_unseen)
            })

            print(f"    ➔ Performance sur {s_test} : R² = {m['R2']:.3f}, RMSE = {m['RMSE']:.2f} µg/m³")

    df_results_axe2 = pd.DataFrame(transfer_results)
    out_file = os.path.join(RESULTS_DIR, "leaderboard_axe2_spatial_transfer.csv")
    df_results_axe2.to_csv(out_file, index=False)
    print(f"\n📊 Résultats Axe 2 exportés : {out_file}")
    return df_results_axe2

# -----------------------------------------------------------------------------
# 6. AXE 3 : MODÉLISATION DU TRANSFERT D'INFILTRATION (EXT -> INT)
# -----------------------------------------------------------------------------
def run_axe3_ext_to_int_coupling():
    print("\n" + "=" * 75)
    print("🔹 AXE 3 : MODÉLISATION DU TRANSFERT D'INFILTRATION (P1 EXT ➔ P2 INT)")
    print("=" * 75)

    if not os.path.exists(MASTER_FILE):
        return pd.DataFrame()

    df_master = pd.read_csv(MASTER_FILE)
    df_master['date'] = pd.to_datetime(df_master['date'])
    sites = sorted(df_master['site'].unique())

    coupling_results = []

    for s in sites:
        site_ext = df_master[(df_master['site'] == s) & (df_master['location'] == 'EXT')].set_index('date').sort_index()
        site_int = df_master[(df_master['site'] == s) & (df_master['location'] == 'INT')].set_index('date').sort_index()

        if site_ext.empty or site_int.empty:
            continue

        # Fusion des mesures simultanées
        common_idx = site_ext.index.intersection(site_int.index)
        if len(common_idx) < 30:
            continue

        df_coupled = pd.DataFrame(index=common_idx)
        df_coupled['PM10_EXT'] = site_ext.loc[common_idx, 'PM10']
        df_coupled['PM10_INT'] = site_int.loc[common_idx, 'PM10']
        df_coupled['Ratio_Infiltration'] = df_coupled['PM10_INT'] / (df_coupled['PM10_EXT'] + 1e-5)

        # Lags et descripteurs de pénétration
        df_coupled['PM10_INT_lag1'] = df_coupled['PM10_INT'].shift(1)
        df_coupled['PM10_EXT_lag1'] = df_coupled['PM10_EXT'].shift(1)

        if 'temp_c' in site_ext.columns and 'temp_c' in site_int.columns:
            df_coupled['Delta_T_EXT_INT'] = site_ext.loc[common_idx, 'temp_c'] - site_int.loc[common_idx, 'temp_c']

        df_coupled.dropna(inplace=True)

        features = [c for c in ['PM10_EXT', 'PM10_EXT_lag1', 'PM10_INT_lag1', 'Delta_T_EXT_INT'] if c in df_coupled.columns]

        # Découpage 80/10/10
        n = len(df_coupled)
        n_train, n_val = int(n * 0.80), int(n * 0.90)

        X_train, y_train = df_coupled.iloc[:n_train][features], df_coupled.iloc[:n_train]['PM10_INT']
        X_test, y_test = df_coupled.iloc[n_val:][features], df_coupled.iloc[n_val:]['PM10_INT']

        if len(X_test) < 3:
            continue

        # Modèle de transfert
        rf_couple = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        rf_couple.fit(X_train, y_train)
        preds = rf_couple.predict(X_test)

        m = compute_metrics(y_test, preds)
        mean_ratio = df_coupled['Ratio_Infiltration'].mean()

        coupling_results.append({
            'Site': s,
            'Ratio_Moyen_Infiltration (INT/EXT)': round(mean_ratio, 2),
            'R² (Prédiction INT via EXT)': round(m['R2'], 3),
            'RMSE (µg/m³)': round(m['RMSE'], 2),
            'MAE (µg/m³)': round(m['MAE'], 2),
            'N_Test': len(y_test)
        })

        print(f"  • {s} ➔ Ratio Infiltration = {mean_ratio:.2f} | R² Modèle Couplé = {m['R2']:.3f} (RMSE = {m['RMSE']:.2f})")

    df_results_axe3 = pd.DataFrame(coupling_results)
    out_file = os.path.join(RESULTS_DIR, "leaderboard_axe3_ext_to_int_coupling.csv")
    df_results_axe3.to_csv(out_file, index=False)
    print(f"\n📊 Résultats Axe 3 exportés : {out_file}")
    return df_results_axe3

# -----------------------------------------------------------------------------
# EXÉCUTION DU PROTOCOLE COMPLET & SÉRIALISATION GLOBALE
# -----------------------------------------------------------------------------
def run_all():
    print("=" * 75)
    print(" LANCEMENT DU BENCHMARK SCIENTIFIQUE COMPLET MULTI-SITES (CADRE LAMTO)")
    print("=" * 75)

    df1, artifacts = run_axe1_local_benchmarks()
    df2 = run_axe2_spatial_transfer()
    df3 = run_axe3_ext_to_int_coupling()

    # Sauvegarde de tous les artefacts pour l'interface Streamlit
    benchmark_bundle = {
        'axe1_df': df1,
        'axe2_df': df2,
        'axe3_df': df3,
        'local_artifacts': artifacts
    }
    save_bundle_path = os.path.join(MODELS_DIR, "benchmark_artifacts.joblib")
    joblib.dump(benchmark_bundle, save_bundle_path)

    print("\n" + "=" * 75)
    print("✅ PROTOCOLE SCIENTIFIQUE EXÉCUTÉ AVEC SUCCÈS !")
    print(f"📦 Tous les modèles et classements sont sauvegardés sous : {save_bundle_path}")
    print("=" * 75)

if __name__ == "__main__":
    run_all()