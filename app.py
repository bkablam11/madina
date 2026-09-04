#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plateforme Décisionnelle & Épidémiologique Qualité de l'Air (PM10 / PM2.5 -

INT / EXT) 100% basée sur les données réelles consolidées.
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

# Configuration de l'affichage
st.set_page_config(
    page_title="Air & Santé - Décision Multi-Sites",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 14px;
        border-left: 4px solid #38bdf8;
    }
</style>
""",
    unsafe_allow_html=True,
)

PATHOLOGIES = [
    "Sore_throat",
    "asthma",
    "sinusitis",
    "acute_respiratory_infections",
    "acute_bronchitis",
]
PATHOLOGY_LABELS = {
    "Sore_throat": "Mal de gorge",
    "asthma": "Asthme",
    "sinusitis": "Sinusite",
    "acute_respiratory_infections": "Infections Respiratoires (IRA)",
    "acute_bronchitis": "Bronchite Aiguë",
}


# ==============================================================================
# 1. CHARGEMENT SÉCURISÉ DES DONNÉES RÉELLES
# ==============================================================================
@st.cache_data
def load_master_dataset():
    master_file = "./processed_data/master_air_health_dataset.csv"
    if not os.path.exists(master_file):
        st.error(
            f"Fichier '{master_file}' introuvable. Veuillez exécuter `python process_and_analyze_health_air.py` d'abord."
        )
        return pd.DataFrame()

    df = pd.read_csv(master_file)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["Mois_Annee"] = df["date"].dt.strftime("%b %Y")
        df = df.sort_values(by="date").reset_index(drop=True)
    return df


df_master = load_master_dataset()

# ==============================================================================
# 2. BARRE LATÉRALE ET FILTRES DYNAMIQUES
# ==============================================================================
st.sidebar.markdown("## ⚙️ Contrôle du Réseau")
st.sidebar.info("📅 **Données :** Série Mensuelle Réelle")

if not df_master.empty:
    sites_disponibles = sorted(df_master["site"].unique())
    selected_site = st.sidebar.selectbox(
        "📍 Station / Quartier :", ["Tous les Sites"] + sites_disponibles
    )
else:
    selected_site = "Tous les Sites"

st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "📑 Modules d'Analyse :",
    [
        "📊 1. Surveillance & Diagnostic Polluants",
        "🚪 2. Infiltration & Salubrité Bâtiment (INT vs EXT)",
        "🫁 3. Impact Sanitaire (Les 5 Pathologies)",
        "📈 4. Modélisation & Sensibilité Risque",
        "🎯 5. Aide à la Décision & Alertes",
    ],
)

# Application stricte du filtre
if selected_site != "Tous les Sites" and not df_master.empty:
    df_current = df_master[df_master["site"] == selected_site].copy()
else:
    df_current = df_master.copy()

# ==============================================================================
# FONCTIONS UTILITAIRES DE CALCUL EN TEMPS RÉEL (SANS FICHIERS STATIQUES)
# ==============================================================================


def calculate_dynamic_stats(df):
    """Calcule les statistiques descriptives à la volée sur le jeu filtré."""
    records = []
    for loc in ["EXT", "INT"]:
        sub = df[df["location"] == loc]
        if sub.empty:
            continue
        row = {
            "Point de mesure": "Extérieur (EXT)"
            if loc == "EXT"
            else "Intérieur (INT)",
            "Observations": len(sub),
            "PM10 Moyen (µg/m³)": round(sub["PM10"].mean(), 1)
            if "PM10" in sub
            else "-",
            "PM10 Min": round(sub["PM10"].min(), 1) if "PM10" in sub else "-",
            "PM10 Max": round(sub["PM10"].max(), 1) if "PM10" in sub else "-",
            "PM2.5 Moyen (µg/m³)": round(sub["PM2.5"].mean(), 1)
            if "PM2.5" in sub
            else "-",
            "PM2.5 Max": round(sub["PM2.5"].max(), 1)
            if "PM2.5" in sub
            else "-",
            "Temp (°C)": round(sub["temp_c"].mean(), 1)
            if "temp_c" in sub
            else "-",
            "Humidité (%)": round(sub["humidity"].mean(), 1)
            if "humidity" in sub
            else "-",
        }
        for p in PATHOLOGIES:
            if p in sub:
                row[f"{PATHOLOGY_LABELS.get(p, p)} (Moy)"] = round(
                    sub[p].mean(), 1
                )
        records.append(row)
    return pd.DataFrame(records)


def compute_paired_ext_int(df):
    """Couple les mesures EXT et INT sur les dates communes pour le ratio I/O."""
    ext = df[df["location"] == "EXT"][["date", "Mois_Annee", "PM10", "PM2.5"]]
    interior = df[df["location"] == "INT"][
        ["date", "Mois_Annee", "PM10", "PM2.5"]
    ]
    merged = pd.merge(
        ext, interior, on=["date", "Mois_Annee"], suffixes=("_EXT", "_INT")
    )
    if not merged.empty:
        merged["Ratio_IO_PM10"] = merged["PM10_INT"] / (
            merged["PM10_EXT"] + 1e-4
        )
        merged["Delta_PM10"] = (
            merged["PM10_EXT"] - merged["PM10_INT"]
        )  # Positif = Bâti protecteur
    return merged


# ==============================================================================
# MODULE 1 : SURVEILLANCE & DIAGNOSTIC DES POLLUANTS
# ==============================================================================
if menu == "📊 1. Surveillance & Diagnostic Polluants":
    st.title(f"📊 Diagnostic Qualité de l'Air : {selected_site}")
    st.markdown(
        f"Analyse des concentrations réelles mesurées in-situ pour **{selected_site}**."
    )

    if not df_current.empty:
        # Cartes métriques globales
        c1, c2, c3, c4 = st.columns(4)
        pm10_mean = (
            df_current["PM10"].mean() if "PM10" in df_current.columns else 0
        )
        pm25_mean = (
            df_current["PM2.5"].mean() if "PM2.5" in df_current.columns else 0
        )
        t_mean = (
            df_current["temp_c"].mean() if "temp_c" in df_current.columns else 0
        )
        h_mean = (
            df_current["humidity"].mean()
            if "humidity" in df_current.columns
            else 0
        )

        with c1:
            st.metric(
                "PM10 Moyen Réel",
                f"{pm10_mean:.1f} µg/m³",
                delta=f"{pm10_mean - 45.0:+.1f} vs Seuil OMS",
                delta_color="inverse",
            )
        with c2:
            st.metric(
                "PM2.5 Moyen Réel",
                f"{pm25_mean:.1f} µg/m³",
                delta=f"{pm25_mean - 15.0:+.1f} vs Seuil OMS",
                delta_color="inverse",
            )
        with c3:
            st.metric("Température Moyenne", f"{t_mean:.1f} °C")
        with c4:
            st.metric("Humidité Moyenne", f"{h_mean:.1f} %")

        st.markdown("---")

        # Graphiques temporels dynamiques avec axes auto-adaptatifs
        col_g1, col_g2 = st.columns(2)

        max_pm10 = df_current["PM10"].max() * 1.15
        max_pm25 = df_current["PM2.5"].max() * 1.15

        with col_g1:
            st.subheader("Évolution Chronologique des PM10 (EXT vs INT)")
            fig_pm10 = px.bar(
                df_current,
                x="Mois_Annee",
                y="PM10",
                color="location",
                barmode="group",
                color_discrete_map={"EXT": "#2563eb", "INT": "#f97316"},
                labels={
                    "PM10": "PM10 (µg/m³)",
                    "Mois_Annee": "Mois",
                    "location": "Capteur",
                },
                template="plotly_white",
            )
            fig_pm10.add_hline(
                y=45,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text="Seuil OMS 24h (45 µg/m³)",
            )
            fig_pm10.update_yaxes(range=[0, max_pm10], title="PM10 (µg/m³)")
            fig_pm10.update_layout(xaxis_tickangle=-45, legend=dict(y=1.1, orientation="h"))
            st.plotly_chart(fig_pm10, use_container_width=True)

        with col_g2:
            st.subheader("Évolution Chronologique des PM2.5 (EXT vs INT)")
            fig_pm25 = px.bar(
                df_current,
                x="Mois_Annee",
                y="PM2.5",
                color="location",
                barmode="group",
                color_discrete_map={"EXT": "#0284c7", "INT": "#ea580c"},
                labels={
                    "PM2.5": "PM2.5 (µg/m³)",
                    "Mois_Annee": "Mois",
                    "location": "Capteur",
                },
                template="plotly_white",
            )
            fig_pm25.add_hline(
                y=15,
                line_dash="dash",
                line_color="#dc2626",
                annotation_text="Seuil OMS 24h (15 µg/m³)",
            )
            fig_pm25.update_yaxes(range=[0, max_pm25], title="PM2.5 (µg/m³)")
            fig_pm25.update_layout(xaxis_tickangle=-45, legend=dict(y=1.1, orientation="h"))
            st.plotly_chart(fig_pm25, use_container_width=True)

        # Tableau de synthèse dynamique calculé en temps réel
        st.markdown(f"### 📋 Synthèse Statistique Calculée ({selected_site})")
        df_dyn_stats = calculate_dynamic_stats(df_current)
        st.dataframe(df_dyn_stats, use_container_width=True)

# ==============================================================================
# MODULE 2 : INFILTRATION & SALUBRITÉ BÂTIMENT (INT vs EXT)
# ==============================================================================
elif menu == "🚪 2. Infiltration & Salubrité Bâtiment (INT vs EXT)":
    st.title(f"🚪 Diagnostic de Pénétration Bâtiment : {selected_site}")
    st.markdown(
        "Évalue la protection passive offerte par le bâti et détecte les sources internes de pollution."
    )

    paired_df = compute_paired_ext_int(df_current)

    if not paired_df.empty:
        c1, c2, c3 = st.columns(3)
        mean_ratio = paired_df["Ratio_IO_PM10"].mean()
        mean_delta = paired_df["Delta_PM10"].mean()
        months_indoor_worse = (paired_df["Ratio_IO_PM10"] > 1.0).sum()

        with c1:
            st.metric(
                "Ratio I/O Moyen",
                f"{mean_ratio:.2f}",
                help="I/O < 1.0 = Bâtiment protecteur | I/O > 1.0 = Sources intérieures",
            )
        with c2:
            st.metric(
                "Atténuation Moyenne du Bâti",
                f"{mean_delta:+.1f} µg/m³",
                delta="Air Intérieur Plus Sain"
                if mean_delta > 0
                else "Air Intérieur Dégradé",
            )
        with c3:
            st.metric(
                "Mois Critiques (I/O > 1)",
                f"{months_indoor_worse} / {len(paired_df)} mois",
            )

        col_b1, col_b2 = st.columns(2)

        with col_b1:
            st.subheader("Ratio de Pénétration I/O par Mois")
            fig_io = px.bar(
                paired_df,
                x="Mois_Annee",
                y="Ratio_IO_PM10",
                color="Ratio_IO_PM10",
                color_continuous_scale="RdYlGn_r",
                labels={
                    "Ratio_IO_PM10": "Ratio I/O (INT / EXT)",
                    "Mois_Annee": "Mois",
                },
                template="plotly_white",
            )
            fig_io.add_hline(
                y=1.0,
                line_dash="dash",
                line_color="red",
                annotation_text="Seuil Critique (I/O = 1.0)",
            )
            fig_io.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_io, use_container_width=True)

        with col_b2:
            st.subheader("Atténuation Nette Mensuelle (PM EXT - PM INT)")
            # Vert si le bâtiment a protégé (Delta > 0), Rouge s'il a pollué (Delta < 0)
            colors = [
                "#16a34a" if val > 0 else "#dc2626"
                for val in paired_df["Delta_PM10"]
            ]
            fig_delta = go.Figure(
                go.Bar(
                    x=paired_df["Mois_Annee"],
                    y=paired_df["Delta_PM10"],
                    marker_color=colors,
                )
            )
            fig_delta.update_layout(
                xaxis_title="Mois",
                yaxis_title="Atténuation Nette (µg/m³)",
                template="plotly_white",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_delta, use_container_width=True)

        st.dataframe(
            paired_df[
                [
                    "Mois_Annee",
                    "PM10_EXT",
                    "PM10_INT",
                    "Ratio_IO_PM10",
                    "Delta_PM10",
                ]
            ].round(2),
            use_container_width=True,
        )
    else:
        st.warning("Mesures simultanées EXT et INT insuffisantes pour ce site.")

# ==============================================================================
# MODULE 3 : IMPACT SANITAIRE (LES 5 PATHOLOGIES)
# ==============================================================================
elif menu == "🫁 3. Impact Sanitaire (Les 5 Pathologies)":
    st.title(f"🫁 Impact Sanitaire & Corrélations Cliniques : {selected_site}")
    st.markdown(
        "Corrélation directe entre les concentrations de poussières et les cas cliniques recensés."
    )

    dispo_pathos = [p for p in PATHOLOGIES if p in df_current.columns]

    if dispo_pathos:
        sel_patho = st.selectbox(
            "🎯 Sélectionner la pathologie :",
            dispo_pathos,
            format_func=lambda x: PATHOLOGY_LABELS.get(x, x),
        )

        # Double axe décisionnel : Pollution vs Cas de Santé
        st.subheader(
            f"Superposition Temporelle : PM10 vs Cas de {PATHOLOGY_LABELS.get(sel_patho, sel_patho)}"
        )

        df_month_agg = (
            df_current.groupby("Mois_Annee")
            .agg({"PM10": "mean", sel_patho: "mean", "date": "min"})
            .sort_values("date")
            .reset_index()
        )

        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

        fig_dual.add_trace(
            go.Bar(
                x=df_month_agg["Mois_Annee"],
                y=df_month_agg["PM10"],
                name="PM10 Moyen (µg/m³)",
                marker_color="#93c5fd",
                opacity=0.7,
            ),
            secondary_y=False,
        )

        fig_dual.add_trace(
            go.Scatter(
                x=df_month_agg["Mois_Annee"],
                y=df_month_agg[sel_patho],
                name=f"Cas recensés ({PATHOLOGY_LABELS.get(sel_patho, sel_patho)})",
                line=dict(color="#dc2626", width=3),
                mode="lines+markers",
            ),
            secondary_y=True,
        )

        fig_dual.update_layout(
            template="plotly_white",
            hovermode="x unified",
            xaxis_tickangle=-45,
            legend=dict(orientation="h", y=1.1),
        )
        fig_dual.update_yaxes(
            title_text="PM10 (µg/m³)",
            secondary_y=False,
            range=[0, df_month_agg["PM10"].max() * 1.2],
        )
        fig_dual.update_yaxes(
            title_text="Nombre de Cas",
            secondary_y=True,
            range=[0, df_month_agg[sel_patho].max() * 1.25],
        )
        st.plotly_chart(fig_dual, use_container_width=True)

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.subheader("Régression Linéaire : PM10 vs Cas")
            fig_reg = px.scatter(
                df_current,
                x="PM10",
                y=sel_patho,
                color="location",
                trendline="ols",
                labels={
                    "PM10": "PM10 (µg/m³)",
                    sel_patho: "Cas enregistrés",
                    "location": "Capteur",
                },
                template="plotly_white",
            )
            st.plotly_chart(fig_reg, use_container_width=True)

        with col_s2:
            st.subheader("Matrice de Corrélation de Spearman (Données Réelles)")
            cols_corr = [
                c
                for c in ["PM10", "PM2.5", "temp_c", "humidity"] + dispo_pathos
                if c in df_current.columns
            ]
            mat_corr = df_current[cols_corr].corr(method="spearman").round(2)
            fig_mat = px.imshow(
                mat_corr,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                template="plotly_white",
            )
            st.plotly_chart(fig_mat, use_container_width=True)

# ==============================================================================
# MODULE 4 : MODÉLISATION & SENSIBILITÉ DU RISQUE (SUR DONNÉES RÉELLES)
# ==============================================================================
elif menu == "📈 4. Modélisation & Sensibilité Risque":
    st.title("📈 Évaluation Empirique du Risque Respiratoire")
    st.markdown(
        "Pente d'accroissement des pathologies en fonction de l'exposition aux particules."
    )

    risk_list = []
    for pat in PATHOLOGIES:
        if pat in df_current.columns and "PM10" in df_current.columns:
            clean = df_current[["PM10", pat]].dropna()
            if len(clean) >= 4 and clean["PM10"].std() > 0:
                slope, intercept, r_val, p_val, std_err = stats.linregress(
                    clean["PM10"], clean[pat]
                )
                r_spearman, p_spearman = stats.spearmanr(
                    clean["PM10"], clean[pat]
                )
                risk_list.append(
                    {
                        "Pathologie": PATHOLOGY_LABELS.get(pat, pat),
                        "Augmentation par +10 µg/m³ PM10": f"{slope * 10:+.1f} cas",
                        "Corrélation Spearman (R)": round(r_spearman, 3),
                        "Significatif (p < 0.05)": "Oui ✅"
                        if p_spearman < 0.05
                        else "Non ⚠️",
                        "Nbr Observations": len(clean),
                    }
                )

    if risk_list:
        df_sens = pd.DataFrame(risk_list)
        st.dataframe(df_sens, use_container_width=True)

        fig_risk = px.bar(
            df_sens,
            x="Pathologie",
            y="Corrélation Spearman (R)",
            color="Corrélation Spearman (R)",
            color_continuous_scale="Reds",
            title=f"Intensité de l'impact des PM10 sur chaque pathologie ({selected_site})",
            template="plotly_white",
        )
        fig_risk.update_yaxes(range=[0, 1.0])
        st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.info("Données insuffisantes pour ajuster les pentes de sensibilité.")

# ==============================================================================
# MODULE 5 : AIDE À LA DÉCISION & ALERTES SANITAIRES
# ==============================================================================
elif menu == "🎯 5. Aide à la Décision & Alertes":
    st.title("🎯 Outil Décisionnel : Prévision d'Impact & Confinement")
    st.markdown(
        f"Simulateur de gestion de crise basé sur l'historique de **{selected_site}**."
    )

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.subheader("1. Scénario d'Épisode de Pollution")
        sim_pm10 = st.slider(
            "PM10 Extérieur projeté (µg/m³)",
            10.0,
            250.0,
            float(df_current["PM10"].mean())
            if not df_current.empty
            else 60.0,
        )

        paired_df = compute_paired_ext_int(df_current)
        ratio_site = (
            paired_df["Ratio_IO_PM10"].mean() if not paired_df.empty else 0.85
        )

        pm10_in_attendu = sim_pm10 * ratio_site

        st.markdown(f"**Ratio d'infiltration propre au site :** `{ratio_site:.2f}`")
        st.metric(
            "Concentration Intérieure Attendue",
            f"{pm10_in_attendu:.1f} µg/m³",
            delta=f"{pm10_in_attendu - 45.0:+.1f} vs Seuil OMS",
            delta_color="inverse",
        )

    with col_d2:
        st.subheader("2. Recommandations Décisionnelles")

        if sim_pm10 <= 45:
            st.success(
                "🟢 **QUALITÉ FAVORABLE**\n- Aération naturelle sans restriction.\n- Faible risque d'exacerbation d'asthme."
            )
        elif sim_pm10 <= 100:
            st.warning(
                "🟡 **VIGILANCE SANITAIRE (Harmattan Modéré)**\n- Recommander la fermeture des fenêtres aux personnes asthmatiques.\n- Réduire les activités physiques intenses en extérieur."
            )
        else:
            st.error(
                "🔴 **ALERTE CRITIQUE (Épisode Majeur de Poussières)**\n- Confinement des écoles et des personnes vulnérables.\n- Port de masque filtrant conseillé en extérieur.\n- Activation du plan d'urgence dans les centres de santé (IRA et bronchites)."
            )

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=sim_pm10,
                title={"text": "Indice de Risque OMS", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, 200]},
                    "bar": {"color": "#1e3a8a"},
                    "steps": [
                        {"range": [0, 45], "color": "#86efac"},
                        {"range": [45, 100], "color": "#fde047"},
                        {"range": [100, 200], "color": "#fca5a5"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 100,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)