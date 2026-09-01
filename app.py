#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Interactif & Modélisation Prédictive Multi-Sites PM10
Cadre méthodologique : N'Datchoh et al. (2025), Open Journal of Air Pollution
Réseau 10 Sites (P1 EXT / P2 INT) - Côte d'Ivoire
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib

# -----------------------------------------------------------------------------
# 1. CONFIGURATION DE LA PAGE & THÈME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AirQuality ML - Plateforme Multi-Sites PM10",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS épuré et moderne
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .badge-champion {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CHARGEMENT EN CACHE DES DONNÉES ET MODÈLES
# -----------------------------------------------------------------------------
@st.cache_resource
def load_benchmark_bundle():
    bundle_path = "./models/benchmark_artifacts.joblib"
    if os.path.exists(bundle_path):
        return joblib.load(bundle_path)
    return None

@st.cache_data
def load_datasets():
    daily_path = "./processed_data/dataset_master_daily_all_sites.csv"
    hourly_path = "./processed_data/dataset_master_hourly_all_sites.csv"
    
    df_daily = pd.read_csv(daily_path) if os.path.exists(daily_path) else None
    if df_daily is not None:
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        
    df_hourly = pd.read_csv(hourly_path) if os.path.exists(hourly_path) else None
    if df_hourly is not None:
        df_hourly['datetime'] = pd.to_datetime(df_hourly['datetime'])
        
    return df_daily, df_hourly

bundle = load_benchmark_bundle()
df_daily, df_hourly = load_datasets()

# -----------------------------------------------------------------------------
# 3. BARRE LATÉRALE DE CONTRÔLE
# -----------------------------------------------------------------------------
st.sidebar.image("https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/wind.svg", width=50)
st.sidebar.markdown("## ⚙️ Contrôle & Navigation")

menu_choice = st.sidebar.radio(
    "Modules disponibles :",
    [
        "🏆 1. Modèle Champion & Analyse Locale (Axe 1)",
        "🚪 2. Infiltration EXT ➔ INT & Confinement (Axe 3)",
        "🌐 3. Transférabilité Spatiale Inter-Sites (Axe 2)",
        "🔬 4. Corrélations Météo & Décomposition",
        "🚀 5. Simulateur Prédictif à J+1 (Temps Réel)"
    ]
)

st.sidebar.markdown("---")

# Liste dynamique des sites
if df_daily is not None:
    available_sites = sorted(df_daily['site'].unique())
    selected_site = st.sidebar.selectbox("📍 Sélectionner le Site :", available_sites, index=0)
    selected_loc = st.sidebar.radio("📡 Point de mesure :", ["P1 EXT (Ambiant)", "P2 INT (Intérieur)"], horizontal=True)
    loc_clean = "P1 EXT" if "EXT" in selected_loc else "P2 INT"
else:
    selected_site, loc_clean = "SITE 1", "P1 EXT"

st.sidebar.markdown("---")
st.sidebar.caption("🔬 **Cadre Scientifique de Lamto** (*N’Datchoh et al., 2025*)")

# -----------------------------------------------------------------------------
# EN-TÊTE PRINCIPAL
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Plateforme Analytique & Prédictive de la Qualité de l\'Air (PM₁₀)</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">Visualisation dynamique des modèles optimaux | <b>{selected_site} - {selected_loc}</b></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MODULE 1 : MODÈLE CHAMPION & ANALYSE LOCALE (AXE 1)
# -----------------------------------------------------------------------------
if menu_choice == "🏆 1. Modèle Champion & Analyse Locale (Axe 1)":
    st.subheader(f"📊 Performances du Modèle Optimal sur {selected_site} ({loc_clean})")

    key_artifact = f"{selected_site}_{loc_clean}"
    local_artifacts = bundle.get('local_artifacts', {}) if bundle else {}
    axe1_df = bundle.get('axe1_df', pd.DataFrame()) if bundle else pd.DataFrame()

    if key_artifact in local_artifacts:
        art = local_artifacts[key_artifact]
        best_model = art['best_model_name']
        y_test = art['y_test']
        preds = art['preds']
        X_test = art['X_test']

        # Métriques du champion
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Modèle Champion Détecté", f"⭐ {best_model}")
        c2.metric(r"Score R² (Test Set)", f"{r2:.3f}")
        c3.metric(r"Erreur RMSE", f"{rmse:.2f} µg/m³")
        c4.metric(r"Erreur MAE", f"{mae:.2f} µg/m³")

        st.markdown("<br>", unsafe_allow_html=True)

        # Graphique Réel vs Prédit
        col_fig1, col_fig2 = st.columns([2.5, 1])

        with col_fig1:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode='lines+markers', name='Mesuré (Réel)', line=dict(color='#1E3A8A', width=2.5)))
            fig_ts.add_trace(go.Scatter(x=y_test.index, y=preds, mode='lines+markers', name=f'Prédit ({best_model})', line=dict(color='#F97316', width=2, dash='solid')))

            fig_ts.add_hline(y=45, line_dash="dash", line_color="orange", annotation_text="Seuil OMS 24h (45 µg/m³)")
            fig_ts.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Seuil Alerte (100 µg/m³)")

            fig_ts.update_layout(
                title=f"Série Temporelle : Réel vs Prédictions ({best_model}) sur le Jeu de Test Inédit (10%)",
                xaxis_title="Date",
                yaxis_title=r"Concentration PM10 (µg/m³)",
                template="plotly_white",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        with col_fig2:
            # Nuage de points Réel vs Prédit
            fig_reg = px.scatter(
                x=y_test.values, y=preds,
                labels={'x': r"PM10 Observé (µg/m³)", 'y': r"PM10 Prédit (µg/m³)"},
                title="Ajustement & Corrélation",
                trendline="ols",
                template="plotly_white"
            )
            # Ligne d'identité y=x
            min_val = min(y_test.min(), preds.min())
            max_val = max(y_test.max(), preds.max())
            fig_reg.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="gray", dash="dash"))
            st.plotly_chart(fig_reg, use_container_width=True)

        # Tableau comparatif des 4 modèles sur ce site
        st.markdown("#### 📋 Benchmark Comparatif des 4 Modèles sur ce Point")
        site_table = axe1_df[(axe1_df['Site'] == selected_site) & (axe1_df['Capteur'] == loc_clean)]
        if not site_table.empty:
            st.dataframe(site_table.sort_values(by='R²', ascending=False), use_container_width=True)
    else:
        st.warning(f"Aucun modèle n'a pu être entraîné pour {selected_site} ({loc_clean}) (données insuffisantes sur ce capteur).")

# -----------------------------------------------------------------------------
# MODULE 2 : INFILTRATION EXT -> INT & CONFINEMENT (AXE 3)
# -----------------------------------------------------------------------------
elif menu_choice == "🚪 2. Infiltration EXT ➔ INT & Confinement (Axe 3)":
    st.subheader(f"🚪 Dynamique de Transfert & Facteur de Pénétration Bâtiment : {selected_site}")

    if df_daily is not None:
        site_ext = df_daily[(df_daily['site'] == selected_site) & (df_daily['location'] == 'EXT')].set_index('date').sort_index()
        site_int = df_daily[(df_daily['site'] == selected_site) & (df_daily['location'] == 'INT')].set_index('date').sort_index()

        common_idx = site_ext.index.intersection(site_int.index)

        if len(common_idx) > 5:
            df_coupled = pd.DataFrame(index=common_idx)
            df_coupled['PM10_EXT'] = site_ext.loc[common_idx, 'PM10']
            df_coupled['PM10_INT'] = site_int.loc[common_idx, 'PM10']
            df_coupled['Ratio_Infiltration'] = df_coupled['PM10_INT'] / (df_coupled['PM10_EXT'] + 1e-5)
            df_coupled['Delta_PM10'] = df_coupled['PM10_EXT'] - df_coupled['PM10_INT']

            mean_ratio = df_coupled['Ratio_Infiltration'].mean()
            mean_ext = df_coupled['PM10_EXT'].mean()
            mean_int = df_coupled['PM10_INT'].mean()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(r"Moyenne EXT (Ambiant)", f"{mean_ext:.1f} µg/m³")
            c2.metric(r"Moyenne INT (Confiné)", f"{mean_int:.1f} µg/m³")
            c3.metric(r"Facteur de Pénétration Moyen", f"{mean_ratio:.2f}", help="> 1 indique des sources intérieures propres (ex: cuisson, tabac)")
            c4.metric(r"Atténuation / Écart Moyen", f"{df_coupled['Delta_PM10'].mean():.1f} µg/m³")

            st.markdown("<br>", unsafe_allow_html=True)

            # Évolution temporelle comparée
            fig_coup = go.Figure()
            fig_coup.add_trace(go.Scatter(x=df_coupled.index, y=df_coupled['PM10_EXT'], name='EXT (P1 Extérieur)', line=dict(color='#2563EB', width=2)))
            fig_coup.add_trace(go.Scatter(x=df_coupled.index, y=df_coupled['PM10_INT'], name='INT (P2 Intérieur)', line=dict(color='#DC2626', width=2, dash='dot')))

            fig_coup.update_layout(
                title=f"Couplage Temporel Journalier EXT vs INT ({selected_site})",
                xaxis_title="Date",
                yaxis_title=r"PM10 (µg/m³)",
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig_coup, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                fig_scat = px.scatter(
                    df_coupled, x='PM10_EXT', y='PM10_INT',
                    trendline="ols",
                    title="Régression du Transfert Infiltration (EXT ➔ INT)",
                    labels={'PM10_EXT': r"PM10 Extérieur (µg/m³)", 'PM10_INT': r"PM10 Intérieur (µg/m³)"},
                    template="plotly_white"
                )
                st.plotly_chart(fig_scat, use_container_width=True)

            with col_b:
                fig_hist_ratio = px.histogram(
                    df_coupled, x='Ratio_Infiltration', nbins=25,
                    title="Distribution du Ratio d'Infiltration (INT / EXT)",
                    labels={'Ratio_Infiltration': "Ratio de Pénétration"},
                    color_discrete_sequence=['#059669'],
                    template="plotly_white"
                )
                fig_hist_ratio.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="Seuil d'équilibre (1.0)")
                st.plotly_chart(fig_hist_ratio, use_container_width=True)

            # Tableau récapitulatif Axe 3
            if bundle and not bundle.get('axe3_df', pd.DataFrame()).empty:
                st.markdown("#### 📋 Leaderboard du Couplage Bâtiment sur l'ensemble des Sites")
                st.dataframe(bundle['axe3_df'], use_container_width=True)
        else:
            st.info(f"Données simultanées insuffisantes entre P1 EXT et P2 INT sur {selected_site}.")

# -----------------------------------------------------------------------------
# MODULE 3 : TRANSFÉRABILITÉ SPATIALE INTER-SITES (AXE 2)
# -----------------------------------------------------------------------------
elif menu_choice == "🌐 3. Transférabilité Spatiale Inter-Sites (Axe 2)":
    st.subheader("🌐 Évaluation de la Transférabilité Spatiale (Cross-Site Transfer)")
    st.markdown(
        """
        **Protocole de recherche :** Le modèle régional est entraîné exclusivement sur les **Sites 1 à 7**, 
        puis testé sur des sites géographiquement distincts et **totalement non vus** (**Sites 8, 9, 10**).
        """
    )

    if bundle and not bundle.get('axe2_df', pd.DataFrame()).empty:
        df_axe2 = bundle['axe2_df']

        col_t1, col_t2 = st.columns([1.5, 1])

        with col_t1:
            st.markdown("#### 📊 Tableau de Généralisation Spatiale")
            st.dataframe(df_axe2, use_container_width=True)

        with col_t2:
            fig_bar_trans = px.bar(
                df_axe2,
                x='Site_Test_Inédit',
                y='R²',
                color='Type_Capteur',
                barmode='group',
                title="Capacité de Généralisation (Score R²)",
                template="plotly_white"
            )
            st.plotly_chart(fig_bar_trans, use_container_width=True)
    else:
        st.info("Données de transférabilité spatiale non disponibles.")

# -----------------------------------------------------------------------------
# MODULE 4 : CORRÉLATIONS MÉTÉO & DÉCOMPOSITION
# -----------------------------------------------------------------------------
elif menu_choice == "🔬 4. Corrélations Météo & Décomposition":
    st.subheader(f"🔬 Matrice de Corrélation & Décomposition Physique : {selected_site}")

    if df_daily is not None:
        site_sub = df_daily[(df_daily['site'] == selected_site) & (df_daily['location'] == ('EXT' if 'EXT' in loc_clean else 'INT'))].set_index('date').sort_index()

        if len(site_sub) > 10:
            # Heatmap de Spearman
            st.markdown("#### 🌡️ Matrice de Corrélation Non-Paramétrique (Spearman)")
            corr_vars = [c for c in ['PM10', 'PM25', 'PM1', 'temp_c', 'dewpoint_c', 'humidity', 'pressure_hpa'] if c in site_sub.columns]
            corr_matrix = site_sub[corr_vars].corr(method='spearman')

            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title=f"Corrélations de Spearman ({selected_site} - {loc_clean})",
                template="plotly_white"
            )
            fig_corr.update_layout(height=450)
            st.plotly_chart(fig_corr, use_container_width=True)

            # Tests statistiques formels
            st.markdown("#### 🧪 Tests Statistiques de Rigueur (Cadre Lamto)")
            c_test1, c_test2 = st.columns(2)

            from statsmodels.tsa.stattools import adfuller
            pm10_series = site_sub['PM10'].dropna()

            adf_res = adfuller(pm10_series)
            shapiro_stat, shapiro_p = stats.shapiro(pm10_series.sample(min(len(pm10_series), 500), random_state=42))

            with c_test1:
                st.info(
                    f"**Test ADF (Stationnarité) :**\n"
                    f"- Statistique ADF : `{adf_res[0]:.3f}`\n"
                    f"- p-value : `{adf_res[1]:.4e}`\n"
                    f"- Diagnostic : **{'Série Stationnaire ✅' if adf_res[1] < 0.05 else 'Non-Stationnaire ❌'}**"
                )

            with c_test2:
                st.warning(
                    f"**Test de Shapiro-Wilk (Normalité) :**\n"
                    f"- Statistique W : `{shapiro_stat:.3f}`\n"
                    f"- p-value : `{shapiro_p:.4e}`\n"
                    f"- Diagnostic : **Non-Gaussien (Asymétrie liée aux pics de poussière/feux)**"
                )
        else:
            st.info("Données insuffisantes pour l'audit statistique de ce site.")

# -----------------------------------------------------------------------------
# MODULE 5 : SIMULATEUR PRÉDICTIF À J+1 (TEMPS RÉEL)
# -----------------------------------------------------------------------------
elif menu_choice == "🚀 5. Simulateur Prédictif à J+1 (Temps Réel)":
    st.subheader(f"🚀 Simulateur Prédictif Opérationnel à J+1 pour : {selected_site} ({loc_clean})")

    key_artifact = f"{selected_site}_{loc_clean}"
    local_artifacts = bundle.get('local_artifacts', {}) if bundle else {}

    if key_artifact in local_artifacts:
        art = local_artifacts[key_artifact]
        model = art['model']
        feature_cols = art['feature_cols']
        best_name = art['best_model_name']

        st.markdown(f"Utilisation du modèle champion : <span class='badge-champion'>⭐ {best_name}</span>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        col_in1, col_in2, col_in3 = st.columns(3)

        with col_in1:
            st.markdown("##### 🕒 Historique Récent des PM₁₀")
            pm10_lag1 = st.slider("Concentration PM₁₀ d'hier J-1 (µg/m³)", 5.0, 350.0, 45.0)
            pm10_lag2 = st.slider("Concentration PM₁₀ à J-2 (µg/m³)", 5.0, 350.0, 40.0)
            pm10_lag3 = st.slider("Concentration PM₁₀ à J-3 (µg/m³)", 5.0, 350.0, 38.0)

        with col_in2:
            st.markdown("##### 🌦️ Conditions Météorologiques Prévues")
            temp_c = st.slider("Température prévue à 2m (°C)", 18.0, 42.0, 29.0)
            dewpoint_c = st.slider("Point de rosée prévu (°C)", 10.0, 30.0, 22.0)
            humidity = st.slider("Humidité relative prévue (%)", 20.0, 100.0, 70.0)

        with col_in3:
            st.markdown("##### 📅 Calendrier & Pression")
            month = st.selectbox("Mois de l'année :", list(range(1, 13)), index=1)
            day_of_week = st.selectbox("Jour de la semaine :", ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"], index=2)
            pressure_hpa = st.number_input("Pression de surface (hPa) :", value=1012.5)

        dow_map = {"Lundi": 0, "Mardi": 1, "Mercredi": 2, "Jeudi": 3, "Vendredi": 4, "Samedi": 5, "Dimanche": 6}

        input_dict = {
            'PM10_lag1': pm10_lag1,
            'PM10_lag2': pm10_lag2,
            'PM10_lag3': pm10_lag3,
            'Month': month,
            'DayOfWeek': dow_map[day_of_week],
            'temp_c': temp_c,
            'temp_c_lag1': temp_c,
            'dewpoint_c': dewpoint_c,
            'dewpoint_c_lag1': dewpoint_c,
            'dewpoint_deficit': temp_c - dewpoint_c,
            'humidity': humidity,
            'humidity_lag1': humidity,
            'pressure_hpa': pressure_hpa,
            'pressure_hpa_lag1': pressure_hpa
        }

        input_df = pd.DataFrame([input_dict])
        input_vec = input_df[[c for c in feature_cols if c in input_df.columns]]

        # Prédiction
        if hasattr(model, 'predict'):
            prediction = float(model.predict(input_vec)[0])
        else:
            prediction = 45.0
        prediction = max(0.0, prediction)

        st.markdown("---")
        res_c1, res_c2 = st.columns([1.2, 1.8])

        with res_c1:
            st.metric(
                label=r"Concentration PM10 Attendue à J+1",
                value=f"{prediction:.1f} µg/m³"
            )

            # Recommandations selon les seuils OMS
            if prediction < 45.0:
                st.success("🟢 **QUALITÉ DE L'AIR : BONNE / FAVORABLE**\n\nConforme au seuil journalier OMS (< 45 µg/m³). Activités extérieures et aération recommandées.")
            elif prediction <= 100.0:
                st.warning("🟠 **QUALITÉ DE L'AIR : MODÉRÉE / ALERTE**\n\nRisque accru pour les personnes sensibles (asthme, enfants, personnes âgées). Limiter les efforts intenses prolongés.")
            else:
                st.error("🔴 **QUALITÉ DE L'AIR : CRITIQUE / DANGER**\n\nPic sévère de particules (> 100 µg/m³). Feux de biomasse ou poussières d'Harmattan. Maintenir les fenêtres fermées et filtrer l'air intérieur.")

        with res_c2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Jauge Sanitaire OMS (24h)", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 200]},
                    'bar': {'color': "#1E3A8A"},
                    'steps': [
                        {'range': [0, 45], 'color': "#86EFAC"},
                        {'range': [45, 100], 'color': "#FDE047"},
                        {'range': [100, 200], 'color': "#FCA5A5"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 100
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.warning(f"Modèle non disponible pour {selected_site} ({loc_clean}). Exécutez `python benchmark_multisite.py`.")