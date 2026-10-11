# 🌍 Plateforme d'Analyse et de Modélisation Prédictive de la Qualité de l'Air (PM₁₀ / PM₂.₅)
### *Cadre Scientifique de Lamto (Côte d'Ivoire) — Réseau Multi-Sites PurpleAir & Impact Sanitaire*

---

## 📖 1. Introduction et Contexte Scientifique

La pollution de l'air par les particules en suspension ($\text{PM}_{10}$ et $\text{PM}_{2.5}$) représente un enjeu environnemental et sanitaire majeur en Afrique de l'Ouest. Cette région est soumise à des contrastes saisonniers marqués :
- **La Grande Saison Sèche (Novembre à Février)** : caractérisée par le vent d'**Harmattan**, transportant des masses de poussières désertiques depuis le Sahara et le Sahel, couplées aux émissions de feux de biomasse et de circulation urbaine.
- **La Saison des Pluies (Mars à Juillet)** : marquée par les précipitations de mousson qui lessivent l'atmosphère (*rain scavenging*), réduisant drastiquement les concentrations au sol.

Ce projet s'inspire directement du protocole expérimental de l'article scientifique de référence :
> **N’Datchoh et al. (2025)** : *"Using Machine Learning Models to Predict Daily PM10 Concentration in the Wet Savanna of Lamto Station in Côte d’Ivoire"*, *Open Journal of Air Pollution*, 14, 73-100. [DOI: 10.4236/ojap.2025.144006](https://doi.org/10.4236/ojap.2025.144006).

L'objectif de ce projet est de traiter, modéliser et valoriser les mesures issues d'un **réseau de micro-capteurs de haute précision (PurpleAir)** déployés sur **10 sites stratégiques**, chacun équipé de capteurs intérieurs et extérieurs, afin de :
1. Prédire les pics de particules à l'échelle journalière ($J+1$) par Machine Learning.
2. Comprendre la dynamique de pénétration des polluants de l'extérieur vers l'intérieur des bâtiments ($P_1\ \text{EXT} \to P_2\ \text{INT}$).
3. Corréler les niveaux de poussières avec l'incidence de 5 pathologies respiratoires aiguës.

---

## 🏗️ 2. Architecture des Données et du Réseau

Le réseau repose sur **10 sites de mesure**, avec deux points d'échantillonnage par site :
- **`P1 EXT` (Capteur Extérieur)** : mesure l'air ambiant, sous influence directe des conditions météorologiques régionales et des sources urbaines.
- **`P2 INT` (Capteur Intérieur)** : mesure l'air confiné à l'intérieur des bâtiments (exposition humaine directe et salubrité).

Chaque moniteur PurpleAir utilise un double faisceau laser (Canal A et Canal B) enregistrant en continu :
- Particules fines : $\text{PM}_{1.0}$, $\text{PM}_{2.5}$, $\text{PM}_{10}$ ($\mu\text{g/m}^3$)
- Variables thermodynamiques : Température ($T$), Point de rosée ($T_d$), Humidité relative ($RH$), Pression atmosphérique ($P$).

---

## ⚙️ 3. Protocole de Traitement Pas-à-Pas

### Étape 1 : Nettoyage et Harmonisation (`pretraitement_site.py`)
1. **Contrôle Qualité des Capteurs (Dual-Channel)** : 
   Pour éviter tout biais lié à une poussière sur l'optique d'un laser, les concentrations sont calculées par la moyenne arithmétique des deux canaux indépendants :
   $$\text{PM}_{10} = \frac{\text{PM}_{10, A} + \text{PM}_{10, B}}{2}$$
2. **Conversions Physiques** :
   Les données d'origine PurpleAir (exprimées en unités anglo-saxonnes) sont converties en unités métriques internationales :
   $$T_{(^\circ\text{C})} = (T_{(^\circ\text{F})} - 32) \times \frac{5}{9}$$
3. **Filtre Temporel Strict** :
   Toutes les données sont harmonisées à partir du **1er décembre 2025** (`START_DATE = 2025-12-01`), éliminant les périodes de calibration préliminaires.
4. **Agrégations Temporelles Multi-Échelles** :
   - Échelle horaire (`1h`) : pour l'étude des cycles diurnes ($63\,292$ mesures).
   - Échelle journalière (`1d`) : pour la modélisation à $J+1$ ($2\,704$ observations consolidées).

### Étape 2 : Imputation Saisonnière et Feature Engineering
- **Imputation par Médiane Calendaire (Méthode de Weed et al. / Lamto)** :
  Les valeurs manquantes dues à des coupures réseau ne sont pas interpolées linéairement (ce qui écraserait les pics), mais remplacées par la médiane du même jour calendaire sur les années adjacentes.
- **Variables Autorégressives** :
  L'analyse d'autocorrélation (ACF/PACF) de Lamto démontre une mémoire atmosphérique sur 3 jours :
  $$\text{Descripteurs} = \left[ \text{PM}_{10}(t-1), \text{PM}_{10}(t-2), \text{PM}_{10}(t-3) \right]$$
- **Physique Thermodynamique** :
  - **Déficit du point de rosée** : $\Delta T = T_{2m} - T_d$ (indicateur direct du niveau de saturation et de l'humidité relative).
  - **Indice de Pénétration Bâtiment** : $\text{Ratio}_{I/O} = \frac{\text{PM}_{10, \text{INT}}}{\text{PM}_{10, \text{EXT}}}$ et l'atténuation nette $\Delta = \text{PM}_{10, \text{EXT}} - \text{PM}_{10, \text{INT}}$.

### Étape 3 : Benchmark Machine Learning en 3 Axes (`benchmark_multisite.py`)
Le pipeline teste 4 architectures avec un découpage chronologique strict sans aucun mélange (**80% Train / 10% Validation / 10% Test**) :
1. **SARIMAX** : modèle autorégressif $(1,0,2) \times (2,0,2)_{12}$ avec variables exogènes.
2. **Random Forest** ($100$ arbres, profondeur max $10$) — *Modèle champion de l'étude de Lamto*.
3. **XGBoost** ($100$ estimateurs, taux d'apprentissage $0.1$, profondeur max $3$).
4. **LightGBM** (boosting `gbdt`, $100$ itérations).

Le protocole explore 3 axes scientifiques majeurs :
- **Axe 1 (Modèles Locaux Dédiés)** : identification du modèle optimal pour chaque capteur individuel.
- **Axe 2 (Transférabilité Spatiale Inter-Sites)** : modèle entraîné sur les Sites 1 à 7 et testé sur les Sites 8 à 10 pour évaluer la capacité à prédire sur un quartier sans capteur.
- **Axe 3 (Couplage Physique EXT $\to$ INT)** : prédiction de l'air confiné intérieur uniquement à partir des données extérieures ambiantes.

---

## 🖥️ 4. L'Application Web Interactive (`app.py`)

L'application Streamlit permet l'exploration visuelle et décisionnelle en 5 modules :
1. **Surveillance & Diagnostic Polluants** : comparaison chronologique EXT vs INT et calcul du Master consolidé des 10 sites sans sommation artificielle.
2. **Infiltration & Salubrité Bâtiment** : évaluation du ratio I/O ($<1.0$ = bâtiment protecteur, $>1.0$ = confinement pollué).
3. **Impact Sanitaire (5 Pathologies)** : corrélation avec Mal de gorge, Asthme, Sinusite, Infections Respiratoires Aiguës (IRA) et Bronchite.
4. **Modélisation & Sensibilité Risque** : régression statistique de l'augmentation des cas par palier de $+10\ \mu\text{g/m}^3$ de $\text{PM}_{10}$.
5. **Aide à la Décision & Alertes OMS** : simulateur prédictif avec jauge tricolore basée sur les seuils 24h de l'OMS :
   - 🟢 **Vert ($<45\ \mu\text{g/m}^3$)** : Qualité favorable, aération recommandée.
   - 🟡 **Jaune ($45 - 100\ \mu\text{g/m}^3$)** : Vigilance, limiter les efforts pour personnes sensibles.
   - 🔴 **Rouge ($>100\ \mu\text{g/m}^3$)** : Alerte critique (Harmattan/feux), confinement et calfeutrage préconisés.

---

## 🚀 5. Guide de Démarrage et Déploiement

### Prérequis
- Python 3.11+
- Le gestionnaire de paquets `uv` (recommandé pour sa rapidité)

### 1. Installation de l'environnement
```bash
# Créer et activer l'environnement
uv venv
source .venv/bin/activate

# Installer les dépendances
uv pip install -r requirements.txt