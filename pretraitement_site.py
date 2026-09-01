#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Automatisé de Prétraitement et Harmonisation Multi-Sites PurpleAir
Conforme au protocole scientifique d'analyse de la qualité de l'air (Cadre Lamto)

Filtre temporel strict : Données prises en compte à partir du 01/12/2025
Génération automatique des jeux de données Journaliers (Daily) et Horaires (Hourly)
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# 1. CONFIGURATION GLOBALE
# -------------------------------------------------------------------------
# Détection automatique du répertoire racine (data ou data07072025)
BASE_DATA_DIR = "./data07072025" if os.path.exists("./data07072025") else "./data"
OUTPUT_DIR = "./processed_data"   # Répertoire de sortie

# 🎯 DATE DE DÉBUT DE CONSIDÉRATION DES DONNÉES (1er Décembre 2025)
START_DATE = pd.Timestamp("2025-12-01 00:00:00")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "by_site"), exist_ok=True)

LINE_HEADER = [
    "UTCDateTime", "mac_address", "firmware_ver", "hardware", "current_temp_f",
    "current_humidity", "current_dewpoint_f", "pressure", "adc", "mem", "rssi",
    "uptime", "pm1_0_cf_1", "pm2_5_cf_1", "pm10_0_cf_1", "pm1_0_atm", "pm2_5_atm",
    "pm10_0_atm", "pm2.5_aqi_cf_1", "pm2.5_aqi_atm", "p_0_3_um", "p_0_5_um",
    "p_1_0_um", "p_2_5_um", "p_5_0_um", "p_10_0_um", "pm1_0_cf_1_b", "pm2_5_cf_1_b",
    "pm10_0_cf_1_b", "pm1_0_atm_b", "pm2_5_atm_b", "pm10_0_atm_b", "pm2.5_aqi_cf_1_b",
    "pm2.5_aqi_atm_b", "p_0_3_um_b", "p_0_5_um_b", "p_1_0_um_b", "p_2_5_um_b",
    "p_5_0_um_b", "p_10_0_um_b", "gas"
]

# -------------------------------------------------------------------------
# 2. FONCTIONS DE PARSING ET CONVERSION
# -------------------------------------------------------------------------
def parse_purpleair_date(date_series):
    """Conversion vectorisée et robuste des dates."""
    clean_series = date_series.astype(str).str.strip()
    parsed = pd.to_datetime(clean_series, format="%Y/%m/%dT%H:%M:%Sz", errors='coerce')
    if parsed.isna().any():
        fallback = pd.to_datetime(clean_series, errors='coerce')
        parsed = parsed.fillna(fallback)
    return parsed

def fahrenheit_to_celsius(f_val):
    """Conversion °F -> °C."""
    return (f_val - 32.0) * (5.0 / 9.0)

# -------------------------------------------------------------------------
# 3. TRAITEMENT D'UN RÉPERTOIRE CAPTEUR
# -------------------------------------------------------------------------
def process_sensor_folder(folder_path, site_name, location_type):
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        return None, None

    print(f"  [+] Traitement de {site_name} | {location_type} ({len(csv_files)} fichiers)...")
    dfs = []

    for file_path in csv_files:
        try:
            base_name = os.path.basename(file_path).split('.')[0]
            if len(base_name) == 8 and base_name.isdigit():
                file_date = pd.to_datetime(base_name, format='%Y%m%d', errors='coerce')
                if pd.notna(file_date) and file_date < START_DATE.floor('D'):
                    continue

            with open(file_path, 'r', encoding='latin1', errors='replace') as f:
                first_line = f.readline()
                if not first_line:
                    continue

            has_header = 'UTCDateTime' in first_line

            if has_header:
                df = pd.read_csv(
                    file_path,
                    sep=",",
                    encoding='latin1',
                    on_bad_lines='skip',
                    engine='c'
                )
            else:
                df = pd.read_csv(
                    file_path,
                    sep=",",
                    header=None,
                    names=LINE_HEADER,
                    encoding='latin1',
                    on_bad_lines='skip',
                    engine='c'
                )

            if df.empty or 'UTCDateTime' not in df.columns:
                continue

            # Parsing des dates
            df['datetime'] = parse_purpleair_date(df['UTCDateTime'])
            df = df.dropna(subset=['datetime'])

            # 🎯 FILTRAGE TEMPOREL STRICT (>= 01/12/2025)
            df = df[df['datetime'] >= START_DATE]
            if df.empty:
                continue

            target_cols = [
                'datetime', 'current_temp_f', 'current_humidity', 'current_dewpoint_f', 'pressure',
                'pm1_0_atm', 'pm1_0_atm_b',
                'pm2_5_atm', 'pm2_5_atm_b',
                'pm10_0_atm', 'pm10_0_atm_b'
            ]
            
            existing_cols = [c for c in target_cols if c in df.columns]
            df_sub = df[existing_cols].copy()

            for c in existing_cols:
                if c != 'datetime':
                    df_sub[c] = pd.to_numeric(df_sub[c], errors='coerce')

            dfs.append(df_sub)

        except Exception as e:
            print(f"    [!] Fichier ignoré ({os.path.basename(file_path)}) : {e}")

    if not dfs:
        return None, None

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values('datetime').drop_duplicates(subset=['datetime']).set_index('datetime')

    # Double vérification du filtre sur l'index fusionné
    merged = merged[merged.index >= START_DATE]
    if merged.empty:
        return None, None

    # Conversions météo
    if 'current_temp_f' in merged.columns:
        merged['temp_c'] = fahrenheit_to_celsius(merged['current_temp_f'])
    if 'current_dewpoint_f' in merged.columns:
        merged['dewpoint_c'] = fahrenheit_to_celsius(merged['current_dewpoint_f'])
    if 'current_humidity' in merged.columns:
        merged['humidity'] = merged['current_humidity']
    if 'pressure' in merged.columns:
        merged['pressure_hpa'] = merged['pressure']

    # Moyennes des canaux A & B
    if 'pm10_0_atm' in merged.columns and 'pm10_0_atm_b' in merged.columns:
        merged['PM10'] = merged[['pm10_0_atm', 'pm10_0_atm_b']].mean(axis=1)
    elif 'pm10_0_atm' in merged.columns:
        merged['PM10'] = merged['pm10_0_atm']

    if 'pm2_5_atm' in merged.columns and 'pm2_5_atm_b' in merged.columns:
        merged['PM25'] = merged[['pm2_5_atm', 'pm2_5_atm_b']].mean(axis=1)
    elif 'pm2_5_atm' in merged.columns:
        merged['PM25'] = merged['pm2_5_atm']

    if 'pm1_0_atm' in merged.columns and 'pm1_0_atm_b' in merged.columns:
        merged['PM1'] = merged[['pm1_0_atm', 'pm1_0_atm_b']].mean(axis=1)

    keep_vars = [c for c in ['PM10', 'PM25', 'PM1', 'temp_c', 'dewpoint_c', 'humidity', 'pressure_hpa'] if c in merged.columns]
    merged_clean = merged[keep_vars]

    # Agrégations temporelles (syntaxe compatible '1h' et '1d')
    hourly_df = merged_clean.resample('1h').mean().dropna(how='all')
    daily_df = merged_clean.resample('1d').mean().dropna(how='all')

    hourly_df['site'] = site_name
    hourly_df['location'] = location_type
    daily_df['site'] = site_name
    daily_df['location'] = location_type

    return hourly_df, daily_df

# -------------------------------------------------------------------------
# 4. SCAN PRINCIPAL MULTI-SITES & CONSOLIDATION GLOBALE
# -------------------------------------------------------------------------
def run_full_pipeline():
    print("=" * 70)
    print(f" SCAN & PRÉTRAITEMENT AUTOMATISÉ (Données >= {START_DATE.strftime('%d/%m/%Y')})")
    print(f" Répertoire source : {BASE_DATA_DIR}")
    print("=" * 70)

    site_folders = sorted(glob.glob(os.path.join(BASE_DATA_DIR, "*")))
    site_folders = [f for f in site_folders if os.path.isdir(f) and "SITE" in os.path.basename(f).upper()]

    all_daily = []
    all_hourly = []

    for site_path in site_folders:
        site_name = os.path.basename(site_path).strip()
        print(f"\n📂 Exploration : {site_name}")

        sub_dirs = glob.glob(os.path.join(site_path, "*"))
        for sub_dir in sub_dirs:
            if not os.path.isdir(sub_dir):
                continue
            sub_name = os.path.basename(sub_dir).strip().upper()
            location_type = "EXT" if ("EXT" in sub_name or "P1" in sub_name) else "INT"
            prefix = "P1_EXT" if location_type == "EXT" else "P2_INT"

            h_df, d_df = process_sensor_folder(sub_dir, site_name, location_type)

            if d_df is not None and not d_df.empty:
                site_slug = site_name.replace(" ", "_")
                d_df.index.name = "date"
                h_df.index.name = "datetime"
                
                # Fichiers individuels par site
                d_df.to_csv(os.path.join(OUTPUT_DIR, "by_site", f"{site_slug}_{prefix}_daily.csv"))
                h_df.to_csv(os.path.join(OUTPUT_DIR, "by_site", f"{site_slug}_{prefix}_hourly.csv"))

                all_daily.append(d_df.reset_index())
                all_hourly.append(h_df.reset_index())

    # Consolidation globale (Journalière et Horaire)
    if all_daily and all_hourly:
        # 1. Dataset consolidé Daily
        master_daily = pd.concat(all_daily, ignore_index=True)
        if 'index' in master_daily.columns:
            master_daily.rename(columns={'index': 'date'}, inplace=True)
        elif 'datetime' in master_daily.columns:
            master_daily.rename(columns={'datetime': 'date'}, inplace=True)

        master_daily_file = os.path.join(OUTPUT_DIR, "dataset_master_daily_all_sites.csv")
        master_daily.to_csv(master_daily_file, index=False)

        # 2. Dataset consolidé Hourly
        master_hourly = pd.concat(all_hourly, ignore_index=True)
        if 'index' in master_hourly.columns:
            master_hourly.rename(columns={'index': 'datetime'}, inplace=True)

        master_hourly_file = os.path.join(OUTPUT_DIR, "dataset_master_hourly_all_sites.csv")
        master_hourly.to_csv(master_hourly_file, index=False)

        print("\n" + "=" * 70)
        print("✅ TRAITEMENT ET CONSOLIDATION TERMINÉS AVEC SUCCÈS !")
        print(f"📊 Fichier consolidé Journalier (Daily) : {master_daily_file}")
        print(f"   • Total observations journalières : {len(master_daily)}")
        print(f"   • Plage de dates : du {master_daily['date'].min()} au {master_daily['date'].max()}")
        print("-" * 70)
        print(f"⏱️ Fichier consolidé Horaire (Hourly)     : {master_hourly_file}")
        print(f"   • Total observations horaires      : {len(master_hourly)}")
        print(f"   • Plage temporelle : du {master_hourly['datetime'].min()} au {master_hourly['datetime'].max()}")
        print("=" * 70)
    else:
        print(f"\n[!] Aucune donnée trouvée après le {START_DATE.strftime('%d/%m/%Y')}.")

if __name__ == "__main__":
    run_full_pipeline()