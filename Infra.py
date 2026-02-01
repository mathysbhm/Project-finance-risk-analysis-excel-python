import os
import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Localisation du fichier
file_name = "Modele_infra.xlsx"
file_path = os.path.join(file_name)

# Lecture
try:
    # On lit le fichier SANS header pour maîtriser les index .iloc
    df = pd.read_excel(file_path, header=None)
    
    # Nettoyage : On remplace les cellules vides ou erreurs par 0 pour éviter les NaNs
    df = df.fillna(0)
    
    # EXTRACTION
    # Dans Excel, la ligne 17 devient l'index 16. La colonne D devient l'index 3 avec iloc
    total_capex   = float(df.iloc[5, 3])   # D6
    prod_annuelle = float(df.iloc[10, 3])  # D11
    prix_base     = float(df.iloc[11, 3])  # D12
    opex_unitaire = float(df.iloc[12, 3])  # D13
    maturite      = int(float(df.iloc[16, 3])) if df.iloc[16, 3] != 0 else 10 # D17
    
    print(f" Données chargées : Capex={total_capex}k€, Maturité={maturite}ans")

except Exception as e:
    print(f" Erreur de lecture : {e}. Utilisation de valeurs par défaut.")
    total_capex, prod_annuelle, prix_base, opex_unitaire, maturite = 200000, 1000000, 60, 15, 10

# PARAMÈTRES FINANCIERS 
invest_initial = -(total_capex * 0.30)
service_dette_annuel = 21000 
n_simulations = 10000
volatilite = 0.20

# SIMULATION DE MONTE CARLO
np.random.seed(42)
scenarios_prix = np.random.lognormal(np.log(prix_base), volatilite, n_simulations)

resultats_tri = []
resultats_dscr = []

for prix in scenarios_prix:
    # On évite les prix aberrants
    prix = max(prix, 1) 
    
    # Waterfall de flux (k€)
    revenus = (prod_annuelle * prix) / 1000
    opex = (prod_annuelle * opex_unitaire) / 1000
    ebitda = revenus - opex
    cfads = ebitda - (max(0, (ebitda - 20000)) * 0.25) # Simplification IS
    
    dscr = cfads / service_dette_annuel
    cash_flow_equity = cfads - service_dette_annuel
    
    # Calcul du TRI (IRR)
    flux = [invest_initial] + [cash_flow_equity] * maturite
    
    #On ne calcule le TRI que si les flux sont cohérents
    tri = npf.irr(flux)
    if not np.isnan(tri):
        resultats_tri.append(tri)
        resultats_dscr.append(dscr)

# VISUALISATION
plt.figure(figsize=(10, 5))
sns.histplot(resultats_tri, kde=True, color="blue")
plt.title(f"Simulation Monte Carlo : TRI Actionnaire (Moyenne: {np.mean(resultats_tri):.1%})")
plt.xlabel("TRI")
plt.show()

print(f"Analyse terminée. Probabilité DSCR < 1.20 : {np.mean(np.array(resultats_dscr) < 1.20):.2%}")

def export_to_pdf(resultats_tri, resultats_dscr, filename="Risk_Report_Infra.pdf"):
    # CALCULS DES KPIS PRINCIPAUX POUR LA CONCLUSION
    tri_moyen = np.mean(resultats_tri)
    prob_defaut = np.mean(np.array(resultats_dscr) < 1.20)
    
    # Logique de diagnostic
    verdict = "VIABLE" if (tri_moyen > 0.15 and prob_defaut < 0.10) else "SOUS CONDITIONS"
    
    # GÉNÉRATION DU PDF
    with PdfPages(filename) as pdf:
        # Création d'une figure au format A4
        fig = plt.figure(figsize=(8.27, 11.69)) 
        
        # 1. Titre Principal
        plt.suptitle("REPORTING D'ANALYSE DE RISQUE\nProjet Infrastructure - Simulation Monte Carlo", 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # 2. Graphique du TRI (Haut)
        ax1 = plt.subplot(3, 1, 1)
        sns.histplot(resultats_tri, kde=True, color="royalblue", ax=ax1)
        ax1.set_title(f"Distribution du TRI (Moyenne : {tri_moyen:.2%})")
        ax1.set_xlabel("TRI (%)")

        # 3. Graphique du DSCR (Milieu)
        ax2 = plt.subplot(3, 1, 2)
        sns.histplot(resultats_dscr, kde=True, color="seagreen", ax=ax2)
        ax2.axvline(1.20, color='red', linestyle='--', label='Covenant (1.20)')
        ax2.set_title(f"Analyse de la Couverture de Dette (DSCR)")
        ax2.set_xlabel("Ratio DSCR")
        ax2.legend()

        # 4. Bloc Conclusion (Bas)
        conclusion_text = (
            f"DIAGNOSTIC FINANCIER :\n\n"
            f"• Rentabilité : Le TRI moyen de {tri_moyen:.2%} confirme la solidité du rendement.\n"
            f"• Risque : La probabilité de rupture de covenant est de {prob_defaut:.2%}.\n"
            f"• Verdict final : PROJET {verdict}.\n\n"
            f"Note : Ce rapport a été généré automatiquement par script Python via une\n"
            f"simulation stochastique de 10 000 itérations sur le prix du baril."
        )

        # Ajout du texte dans un cadre en bas de page
        plt.figtext(0.15, 0.05, conclusion_text, fontsize=11, linespacing=1.8,
                    bbox={"facecolor":"whitesmoke", "alpha":0.8, "pad":10, "edgecolor":"gray"})

        # Ajustement rapide de l'espacement pour que rien ne se chevauche
        plt.tight_layout(rect=[0, 0.15, 1, 0.92])
        
        pdf.savefig(fig)
        plt.close()
    
    print(f" Rapport complet généré : {filename}")

# Appel final
export_to_pdf(resultats_tri, resultats_dscr)

#Streamlit
st.sidebar.header("Paramètres de la Simulation")
# On les met tout en haut (les) pour qu'ils soient toujours visibles
prix_moyen = st.sidebar.slider("Prix du Baril ($)", 30, 100, 60)
volatilité = st.sidebar.slider("Volatilité du marché (%)", 5, 40, 20) / 100

if st.sidebar.button("Calculer la Viabilité"):
    
    scenarios_prix = np.random.lognormal(
        mean=np.log(prix_moyen), 
        sigma=volatilité, 
        size=10000
    )
    
    # On initialise les listes vides avant la boucle
    resultats_tri = []
    resultats_dscr = []

    for prix in scenarios_prix:
        # On évite les prix aberrants
        prix = max(prix, 1) 
        
        # Waterfall de flux (k€)
        revenus = (prod_annuelle * prix) / 1000
        opex = (prod_annuelle * opex_unitaire) / 1000
        ebitda = revenus - opex
        cfads = ebitda - (max(0, (ebitda - 20000)) * 0.25) # Simplification IS
        
        dscr = cfads / service_dette_annuel
        cash_flow_equity = cfads - service_dette_annuel
        
        # Calcul du TRI (IRR)
        flux = [invest_initial] + [cash_flow_equity] * maturite
        
        # Sécurité : on ne calcule le TRI que si les flux sont cohérents
        tri = npf.irr(flux)
        if not np.isnan(tri):
            resultats_tri.append(tri)
            resultats_dscr.append(dscr)
        
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(resultats_tri, kde=True, color="royalblue", ax=ax)
    st.pyplot(fig)
    
    st.success(f"Simulation terminée : TRI moyen de {np.mean(resultats_tri):.2%}")

