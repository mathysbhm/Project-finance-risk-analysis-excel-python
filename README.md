#Infrastructure Project Finance Analysis - Monte Carlo Simulation
Ce projet présente un outil d'analyse de risque financier conçu pour transformer un modèle Excel statique en une simulation de performance dynamique. L'objectif est d'évaluer la viabilité d'un projet d'infrastructure face à la volatilité des marchés énergétiques.

Fonctionnalités techniques:
Extraction de données : Le script infra.py automatise la lecture et le nettoyage des données financières structurées à partir du fichier source Modele_infra.xlsx.

Simulation de Monte Carlo : Exécution de 10 000 itérations utilisant une distribution log-normale pour modéliser l'incertitude du prix des matières premières.

Analyse de la Dette et Covenants : Calcul du CFADS et monitoring du ratio de couverture de la dette (DSCR) avec détection des probabilités de rupture de covenant au seuil de 1.20.

Interface Décisionnelle : Déploiement d'une application Streamlit permettant de tester en temps réel la sensibilité du TRI Actionnaire (Equity IRR) aux paramètres de marché.

Reporting Automatisé : Génération d'un diagnostic financier exportable en PDF incluant les distributions statistiques et le verdict de viabilité du projet.

Enjeux métier
Ce modèle vise à démontrer une capacité à structurer une waterfall de flux complexe et à automatiser les processus de Due Diligence.
