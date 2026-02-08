#MathysBrahmiaFerrier
#Modélisation de projets


import os
import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
# ===== LECTURE DES DONNÉES =====
base_dir = os.path.dirname(os.path.abspath(__file__))
file_name = "Modele_infra.xlsx"
file_path = os.path.join(base_dir, file_name)

try:
    df = pd.read_excel(file_path, header=None)
    df = df.fillna(0)
    
    # Extraction des paramètres
    total_capex   = float(df.iloc[5, 3])   # D6: 200 000 k€
    prod_annuelle = float(df.iloc[10, 3])  # D11: 1 000 000 barils
    prix_base     = float(df.iloc[11, 3])  # D12: 60 $/bbl
    opex_unitaire = float(df.iloc[12, 3])  # D13: 15 $/bbl
    maturite      = int(float(df.iloc[16, 3])) if df.iloc[16, 3] != 0 else 10 # D17
    
    print(f"✅ Données chargées : Capex={total_capex}k€, Maturité={maturite}ans")

except Exception as e:
    print(f"⚠️ Erreur de lecture : {e}. Utilisation de valeurs par défaut.")
    total_capex, prod_annuelle, prix_base, opex_unitaire, maturite = 200000, 1000000, 60, 15, 10

# ===== PARAMÈTRES FINANCIERS =====
leverage_ratio = 0.70  # 70% dette
equity_ratio = 0.30    # 30% fonds propres
invest_initial = -(total_capex * equity_ratio)  # -60 000 k€

dette_initiale = total_capex * leverage_ratio  # 140 000 k€
taux_interet = 0.05  # 5% selon Excel
n_simulations = 10000
volatilite = 0.25  # Augmenté à 25% (plus réaliste pour le pétrole)

# ===== FONCTION DE CALCUL DU SERVICE DE DETTE =====
def calculer_service_dette(dette_restante, annee, maturite, taux):
    """
    Calcule le service de dette annuel (intérêts + principal)
    Amortissement linéaire du principal sur la maturité
    """
    if annee > maturite or dette_restante <= 0:
        return 0, 0, 0
    
    interets = dette_restante * taux
    remb_principal = dette_initiale / maturite
    service_total = interets + remb_principal
    dette_nouvelle = dette_restante - remb_principal
    
    return service_total, interets, dette_nouvelle

#Simulation de Monte Carlo
np.random.seed(42)
scenarios_prix = np.random.lognormal(np.log(prix_base), volatilite, n_simulations)

resultats_tri = []
resultats_dscr = []
resultats_dscr_min = []  # DSCR minimum sur la période

for prix in scenarios_prix:
    prix = max(prix, 1)  
    
    flux_equity = [invest_initial]  # Année 0
    dette_restante = dette_initiale
    dscr_annuels = []
    
    for annee in range(1, maturite + 1):
        # Revenus et coûts (en k€)
        revenus = (prod_annuelle * prix) / 1000
        opex = (prod_annuelle * opex_unitaire) / 1000
        ebitda = revenus - opex
        
        # Amortissements (simplifié : linéaire sur 10 ans)
        amortissements = total_capex / maturite
        ebit = ebitda - amortissements
        
        # Service de dette
        service_dette, interets, dette_restante = calculer_service_dette(
            dette_restante, annee, maturite, taux_interet
        )
        
        # Impôts (25% sur EBIT après intérêts)
        ebt = ebit - interets
        impots = max(0, ebt * 0.25)
        
        # CFADS (Cash Flow Available for Debt Service)
        cfads = ebitda - impots
        
        # DSCR
        if service_dette > 0:
            dscr = cfads / service_dette
            dscr_annuels.append(dscr)
        
        # Cash flow to equity
        cf_equity = cfads - service_dette
        flux_equity.append(cf_equity)
    
    # Calcul du TRI
    try:
        tri = npf.irr(flux_equity)
        if not np.isnan(tri) and -1 < tri < 5:  # Filtre des valeurs aberrantes
            resultats_tri.append(tri)
            if dscr_annuels:
                resultats_dscr.append(np.mean(dscr_annuels))
                resultats_dscr_min.append(min(dscr_annuels))
    except:
        pass

# ===== STATISTIQUES =====
tri_moyen = np.mean(resultats_tri)
tri_p10 = np.percentile(resultats_tri, 10)
tri_p90 = np.percentile(resultats_tri, 90)
prob_dscr_breach = np.mean(np.array(resultats_dscr_min) < 1.20)

print("\n" + "="*50)
print("📊 RÉSULTATS DE LA SIMULATION")
print("="*50)
print(f"TRI moyen          : {tri_moyen:.2%}")
print(f"TRI P10 / P90      : {tri_p10:.2%} / {tri_p90:.2%}")
print(f"Prob. DSCR < 1.20  : {prob_dscr_breach:.2%}")
print("="*50)

# ===== VISUALISATION =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution du TRI
sns.histplot(resultats_tri, kde=True, color="royalblue", ax=axes[0])
axes[0].axvline(tri_moyen, color='red', linestyle='--', label=f'Moyenne: {tri_moyen:.2%}')
axes[0].axvline(0.15, color='green', linestyle='--', label='Seuil 15%')
axes[0].set_title("Distribution du TRI Equity")
axes[0].set_xlabel("TRI")
axes[0].legend()

# Distribution du DSCR minimum
sns.histplot(resultats_dscr_min, kde=True, color="seagreen", ax=axes[1])
axes[1].axvline(1.20, color='red', linestyle='--', label='Covenant 1.20x')
axes[1].set_title("Distribution du DSCR Minimum")
axes[1].set_xlabel("DSCR")
axes[1].legend()

plt.tight_layout()
plt.savefig("simulation_results.png", dpi=300)
plt.show()

# ===== EXPORT PDF =====
def export_to_pdf(resultats_tri, resultats_dscr_min, filename="Risk_Report_Infra.pdf"):
    tri_moyen = np.mean(resultats_tri)
    prob_defaut = np.mean(np.array(resultats_dscr_min) < 1.20)
    
    verdict = "VIABLE ✅" if (tri_moyen > 0.15 and prob_defaut < 0.10) else "SOUS CONDITIONS ⚠️"
    
    with PdfPages(filename) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        
        plt.suptitle("REPORTING D'ANALYSE DE RISQUE\nProjet Infrastructure - Simulation Monte Carlo", 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Graphique TRI
        ax1 = plt.subplot(3, 1, 1)
        sns.histplot(resultats_tri, kde=True, color="royalblue", ax=ax1)
        ax1.axvline(tri_moyen, color='red', linestyle='--', linewidth=2)
        ax1.axvline(0.15, color='green', linestyle='--', label='Seuil 15%')
        ax1.set_title(f"Distribution du TRI Equity (Moyenne : {tri_moyen:.2%})")
        ax1.set_xlabel("TRI (%)")
        ax1.legend()

        # Graphique DSCR
        ax2 = plt.subplot(3, 1, 2)
        sns.histplot(resultats_dscr_min, kde=True, color="seagreen", ax=ax2)
        ax2.axvline(1.20, color='red', linestyle='--', linewidth=2, label='Covenant (1.20x)')
        ax2.set_title("Analyse de la Couverture de Dette (DSCR Minimum)")
        ax2.set_xlabel("Ratio DSCR")
        ax2.legend()

        # Conclusion
        conclusion_text = (
            f"DIAGNOSTIC FINANCIER :\n\n"
            f"• Rentabilité : Le TRI moyen de {tri_moyen:.2%} {'dépasse' if tri_moyen > 0.15 else 'est inférieur à'} le seuil de 15%.\n"
            f"• Risque : La probabilité de rupture de covenant (DSCR < 1.20) est de {prob_defaut:.2%}.\n"
            f"• Verdict final : PROJET {verdict}\n\n"
            f"Méthodologie : Simulation Monte Carlo de {n_simulations:,} itérations sur le prix du baril\n"
            f"(distribution lognormale, σ={volatilite:.0%}). Service de dette calculé avec\n"
            f"amortissement linéaire du principal sur {maturite} ans."
        )

        plt.figtext(0.15, 0.05, conclusion_text, fontsize=10, linespacing=1.8,
                    bbox={"facecolor":"whitesmoke", "alpha":0.9, "pad":10, "edgecolor":"gray"})

        plt.tight_layout(rect=[0, 0.15, 1, 0.92])
        pdf.savefig(fig)
        plt.close()
    
    print(f"✅ Rapport PDF généré : {filename}")

export_to_pdf(resultats_tri, resultats_dscr_min)

# ===== STREAMLIT APP =====
st.set_page_config(page_title="Project Finance - Monte Carlo", layout="wide")

st.title("Simulation de Financement de Projet - Infrastructure Pétrolière")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("CAPEX Total", f"{total_capex:,.0f} k€")
    st.metric("Leverage", f"{leverage_ratio:.0%}")
with col2:
    st.metric("Production", f"{prod_annuelle:,.0f} barils/an")
    st.metric("OPEX", f"{opex_unitaire} $/baril")
with col3:
    st.metric("Maturité", f"{maturite} ans")
    st.metric("Taux d'intérêt", f"{taux_interet:.1%}")

st.sidebar.header("⚙️ Paramètres de Simulation")
prix_moyen = st.sidebar.slider("Prix du Baril ($)", 30, 100, int(prix_base))
vol_input = st.sidebar.slider("Volatilité du marché (%)", 10, 40, 25)
volatilite_sim = vol_input / 100
n_sim = st.sidebar.number_input("Nombre de simulations", 1000, 50000, 10000, step=1000)

if st.sidebar.button("🚀 Lancer la Simulation", type="primary"):
    with st.spinner("Simulation en cours..."):
        scenarios_prix_sim = np.random.lognormal(np.log(prix_moyen), volatilite_sim, n_sim)
        
        resultats_tri_sim = []
        resultats_dscr_sim = []
        resultats_dscr_min_sim = []

        for prix in scenarios_prix_sim:
            prix = max(prix, 1)
            flux_equity = [invest_initial]
            dette_restante = dette_initiale
            dscr_annuels = []
            
            for annee in range(1, maturite + 1):
                revenus = (prod_annuelle * prix) / 1000
                opex = (prod_annuelle * opex_unitaire) / 1000
                ebitda = revenus - opex
                amortissements = total_capex / maturite
                ebit = ebitda - amortissements
                
                service_dette, interets, dette_restante = calculer_service_dette(
                    dette_restante, annee, maturite, taux_interet
                )
                
                ebt = ebit - interets
                impots = max(0, ebt * 0.25)
                cfads = ebitda - impots
                
                if service_dette > 0:
                    dscr = cfads / service_dette
                    dscr_annuels.append(dscr)
                
                cf_equity = cfads - service_dette
                flux_equity.append(cf_equity)
            
            try:
                tri = npf.irr(flux_equity)
                if not np.isnan(tri) and -1 < tri < 5:
                    resultats_tri_sim.append(tri)
                    if dscr_annuels:
                        resultats_dscr_sim.append(np.mean(dscr_annuels))
                        resultats_dscr_min_sim.append(min(dscr_annuels))
            except:
                pass
        
        # Affichage des résultats
        st.success(f" Simulation terminée : {len(resultats_tri_sim):,} scénarios convergents")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("TRI Moyen", f"{np.mean(resultats_tri_sim):.2%}")
        with col2:
            st.metric("TRI P10", f"{np.percentile(resultats_tri_sim, 10):.2%}")
        with col3:
            prob_breach = np.mean(np.array(resultats_dscr_min_sim) < 1.20)
            st.metric("Prob. Breach DSCR", f"{prob_breach:.2%}", 
                     delta=f"{'⚠️ Élevé' if prob_breach > 0.10 else '✅ Acceptable'}")
        
        # Graphiques
        fig_sim, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.histplot(resultats_tri_sim, kde=True, color="royalblue", ax=axes[0])
        axes[0].axvline(np.mean(resultats_tri_sim), color='red', linestyle='--', linewidth=2)
        axes[0].axvline(0.15, color='green', linestyle='--', label='Seuil 15%')
        axes[0].set_title("Distribution du TRI Equity")
        axes[0].set_xlabel("TRI")
        axes[0].legend()
        
        sns.histplot(resultats_dscr_min_sim, kde=True, color="seagreen", ax=axes[1])
        axes[1].axvline(1.20, color='red', linestyle='--', linewidth=2, label='Covenant')
        axes[1].set_title("Distribution du DSCR Minimum")
        axes[1].set_xlabel("DSCR")
        axes[1].legend()
        
        plt.tight_layout()
        st.pyplot(fig_sim)
        
        # Analyse de sensibilité
        st.markdown("---")
        st.subheader("📈 Analyse de Sensibilité")
        
        prix_range = np.linspace(30, 100, 15)
        tri_sensitivity = []
        
        for p in prix_range:
            flux_test = [invest_initial]
            dette_test = dette_initiale
            
            for annee in range(1, maturite + 1):
                revenus = (prod_annuelle * p) / 1000
                opex = (prod_annuelle * opex_unitaire) / 1000
                ebitda = revenus - opex
                amortissements = total_capex / maturite
                ebit = ebitda - amortissements
                
                service, interets, dette_test = calculer_service_dette(
                    dette_test, annee, maturite, taux_interet
                )
                
                ebt = ebit - interets
                impots = max(0, ebt * 0.25)
                cfads = ebitda - impots
                cf_equity = cfads - service
                flux_test.append(cf_equity)
            
            try:
                tri_test = npf.irr(flux_test)
                tri_sensitivity.append(tri_test if not np.isnan(tri_test) else 0)
            except:
                tri_sensitivity.append(0)
        
        fig_sens, ax = plt.subplots(figsize=(10, 5))
        ax.plot(prix_range, [t*100 for t in tri_sensitivity], linewidth=2, marker='o')
        ax.axhline(15, color='green', linestyle='--', label='Seuil 15%')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel("Prix du baril ($)")
        ax.set_ylabel("TRI (%)")
        ax.set_title("Sensibilité du TRI au prix du pétrole")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig_sens)


