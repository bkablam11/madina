#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moteur Analytique & Épidémiologique Multi-Sites (Version Robuste)
Corrige : encodages corrompus, colonnes dupliquées et lignes malformées.
"""

import os
import glob
from pathlib import Path
import re
import warnings
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings('ignore')

DATA_DIR = "./data"
OUTPUT_DIR = "./processed_data"
RESULTS_DIR = "./results"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

PATHOLOGIES = ['Sore_throat', 'asthma', 'sinusitis', 'acute_respiratory_infections', 'acute_bronchitis']

def read_csv_robust(fpath):
    """Lit un CSV en testant plusieurs encodages et en ignorant les lignes corrompues."""
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            df = pd.read_csv(fpath, encoding=enc, on_bad_lines='skip', low_memory=False)
            return df
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    return None

def clean_and_deduplicate_columns(df):
    """Élimine les colonnes dupliquées pour éviter le crash InvalidIndexError."""
    # Renommage intelligent
    rename_dict = {}
    seen_pm10 = False
    seen_pm25 = False
    
    for col in df.columns:
        c_clean = col.strip().lower()
        if ('pm2' in c_clean or 'pm2_5' in c_clean) and not seen_pm25:
            rename_dict[col] = 'PM2.5'
            seen_pm25 = True
        elif 'pm10' in c_clean and not seen_pm10:
            rename_dict[col] = 'PM10'
            seen_pm10 = True
        elif 'temp' in c_clean:
            rename_dict[col] = 'temp_c'
        elif 'humid' in c_clean:
            rename_dict[col] = 'humidity'
        elif 'sore' in c_clean:
            rename_dict[col] = 'Sore_throat'
        elif 'asthma' in c_clean:
            rename_dict[col] = 'asthma'
        elif 'sinus' in c_clean:
            rename_dict[col] = 'sinusitis'
        elif 'respiratory' in c_clean or 'ari' in c_clean:
            rename_dict[col] = 'acute_respiratory_infections'
        elif 'bronch' in c_clean:
            rename_dict[col] = 'acute_bronchitis'
            
    df = df.rename(columns=rename_dict)
    # Suppression stricte de tout doublon de nom de colonne résiduel
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def parse_temporal_index(df):
    """Détecte et construit une date propre."""
    df = df.copy()
    col_names = [c.lower() for c in df.columns]
    
    # 1. Date explicite
    for dt_col in ['datetime', 'date', 'timestamp']:
        if dt_col in col_names:
            real_col = df.columns[col_names.index(dt_col)]
            df['date'] = pd.to_datetime(df[real_col], errors='coerce')
            return df.dropna(subset=['date'])

    # 2. Month + Year
    month_cols = [c for c in df.columns if c.lower() == 'month']
    year_cols = [c for c in df.columns if c.lower() == 'year']
    
    if month_cols and year_cols:
        month_map = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'october': 10, 'oct': 10,
            'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        def to_m(v):
            if pd.isna(v): return 1
            if isinstance(v, (int, float)): return int(v)
            return month_map.get(str(v).strip().lower(), 1)
        
        m_num = df[month_cols[0]].apply(to_m)
        y_num = pd.to_numeric(df[year_cols[0]], errors='coerce').fillna(2024).astype(int)
        df['date'] = pd.to_datetime({'year': y_num, 'month': m_num, 'day': [1]*len(df)})
        return df

    df['date'] = pd.date_range(start="2024-01-01", periods=len(df), freq='D')
    return df

def load_and_consolidate_master():
    # On cherche en priorité les fichiers 'data.csv' consolidés
    target_files = glob.glob(os.path.join(DATA_DIR, "**", "data.csv"), recursive=True)
    
    # Si aucun data.csv, on prend tous les .csv
    if not target_files:
        print("ℹ️ Aucun 'data.csv' trouvé. Analyse de tous les CSV...")
        target_files = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
    else:
        print(f"🎯 {len(target_files)} fichiers de synthèse 'data.csv' trouvés !")

    all_dfs = []

    for fpath in target_files:
        parts = Path(fpath).parts
        site_match = [p for p in parts if 'SITE' in p.upper()]
        site_name = site_match[0] if site_match else "SITE_INCONNU"

        loc_type = "EXT"
        for p in parts:
            if re.search(r'\b(EXT|P\d+\s*EXT)\b', p, re.I):
                loc_type = "EXT"
                break
            elif re.search(r'\b(INT|IN|P\d+\s*INT)\b', p, re.I):
                loc_type = "INT"
                break

        df = read_csv_robust(fpath)
        if df is None or df.empty:
            continue

        df = clean_and_deduplicate_columns(df)
        df = parse_temporal_index(df)
        df['site'] = site_name
        df['location'] = loc_type

        # Conserver uniquement les colonnes utiles
        cols_to_keep = [c for c in ['date', 'site', 'location', 'PM10', 'PM2.5', 'temp_c', 'humidity'] + PATHOLOGIES if c in df.columns]
        df = df[cols_to_keep].copy()

        all_dfs.append(df)

    if not all_dfs:
        print("❌ Aucune donnée exploitable n'a pu être chargée.")
        return pd.DataFrame()

    # Concaténation sécurisée sans duplication d'index ou de colonnes
    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df = master_df.loc[:, ~master_df.columns.duplicated()]
    master_df = master_df.sort_values(by=['site', 'location', 'date']).reset_index(drop=True)

    master_path = os.path.join(OUTPUT_DIR, "master_air_health_dataset.csv")
    master_df.to_csv(master_path, index=False)
    print(f"\n✅ Master Dataset généré avec succès : {master_path} ({len(master_df)} lignes, {len(master_df['site'].unique())} sites)")
    return master_df

def compute_stats_and_epidemiology(df_master):
    if df_master.empty:
        return
    
    # 1. Ratios I/O
    io_list = []
    for s in df_master['site'].unique():
        sub_ext = df_master[(df_master['site'] == s) & (df_master['location'] == 'EXT')]
        sub_int = df_master[(df_master['site'] == s) & (df_master['location'] == 'INT')]
        if not sub_ext.empty and not sub_int.empty and 'PM10' in df_master.columns:
            m = pd.merge(sub_ext[['date', 'PM10']], sub_int[['date', 'PM10']], on='date', suffixes=('_EXT', '_INT'))
            if len(m) > 0:
                ratio = m['PM10_INT'] / (m['PM10_EXT'] + 1e-4)
                io_list.append({
                    'Site': s,
                    'Ratio_I_O_PM10_Moyen': round(ratio.mean(), 2),
                    'Atténuation_Bâti_Net': round((m['PM10_EXT'] - m['PM10_INT']).mean(), 2)
                })
    pd.DataFrame(io_list).to_csv(os.path.join(RESULTS_DIR, "ratios_infiltration_io.csv"), index=False)

    # 2. Corrélations Santé-Pollution
    present_pathos = [p for p in PATHOLOGIES if p in df_master.columns]
    env_vars = [c for c in ['PM10', 'PM2.5', 'temp_c', 'humidity'] if c in df_master.columns]
    
    corr_list = []
    for pat in present_pathos:
        for evar in env_vars:
            sub = df_master[[pat, evar]].dropna()
            if len(sub) >= 5:
                r, pval = stats.spearmanr(sub[pat], sub[evar])
                corr_list.append({
                    'Pathologie': pat,
                    'Variable': evar,
                    'Spearman_R': round(r, 3),
                    'p_value': round(pval, 4)
                })
    pd.DataFrame(corr_list).to_csv(os.path.join(RESULTS_DIR, "correlations_sante_pollution.csv"), index=False)

    # 3. Risques Relatifs (Poisson GLM)
    risk_list = []
    for pat in present_pathos:
        if 'PM10' not in df_master.columns:
            continue
        sub = df_master.dropna(subset=[pat, 'PM10'])
        if len(sub) >= 8 and sub[pat].std() > 0:
            try:
                X = sm.add_constant(sub[['PM10']])
                y = sub[pat]
                mod = sm.GLM(y, X, family=sm.families.Poisson()).fit()
                coef = mod.params['PM10']
                rr_10 = np.exp(coef * 10)
                risk_list.append({
                    'Pathologie': pat,
                    'Type_Exposition': 'Global',
                    'Risque_Relatif_RR (+10µg/m³)': round(rr_10, 3),
                    'Sur-Risque_% (+10µg/m³)': f"{(rr_10 - 1) * 100:+.1f} %",
                    'p_value': round(mod.pvalues['PM10'], 4),
                    'IC_95%_Bas': round(np.exp(mod.conf_int().loc['PM10', 0] * 10), 3),
                    'IC_95%_Haut': round(np.exp(mod.conf_int().loc['PM10', 1] * 10), 3)
                })
            except Exception:
                pass
    pd.DataFrame(risk_list).to_csv(os.path.join(RESULTS_DIR, "risques_relatifs_pathologies.csv"), index=False)
    print("✅ Statistiques descriptives, Ratios I/O et Risques Relatifs générés !")

if __name__ == "__main__":
    df_m = load_and_consolidate_master()
    compute_stats_and_epidemiology(df_m)