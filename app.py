"""
Drug-Induced Autoimmunity (DIA) Risk Prediction System
Streamlit Deployment Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import json
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem import Draw
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# OPTIMIZED DIA PREDICTOR CLASS (Must be defined before loading model)
# ============================================================================

class OptimizedDIAPredictor:
    """Optimized DIA predictor with custom threshold"""
    def __init__(self, base_model, threshold=0.4):
        self.base_model = base_model
        self.threshold = threshold
    
    def predict(self, X):
        return (self.base_model.predict_proba(X)[:, 1] >= self.threshold).astype(int)
    
    def predict_proba(self, X):
        return self.base_model.predict_proba(X)
    
    def predict_with_risk(self, X):
        proba = self.base_model.predict_proba(X)[:, 1]
        preds = (proba >= self.threshold).astype(int)
        risk_category = np.where(proba < 0.4, 'Low',
                         np.where(proba < 0.6, 'Moderate',
                         np.where(proba < 0.8, 'High', 'Very High')))
        return preds, risk_category, proba

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="DIA Risk Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .risk-moderate {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .risk-high {
        background: linear-gradient(135deg, #fa8231 0%, #ff6b6b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .risk-very-high {
        background: linear-gradient(135deg, #c0392b 0%, #8e44ad 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    try:
        model = joblib.load('optimized_dia_predictor_threshold_0.4.pkl')
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure 'optimized_dia_predictor_threshold_0.4.pkl' is in the directory.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

@st.cache_data
def load_feature_importance():
    """Load feature importance data"""
    try:
        importance_df = pd.read_csv('comprehensive_feature_importance.csv')
        return importance_df.head(20)
    except:
        return None

def calculate_all_rdkit_descriptors(smiles):
    """Calculate comprehensive RDKit descriptors from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES string"
        
        mol = Chem.AddHs(mol)
        descriptors = {}
        
        # Basic descriptors
        descriptors['BalabanJ'] = rdMolDescriptors.CalcBalabanJ(mol)
        descriptors['BertzCT'] = rdMolDescriptors.CalcBertzCT(mol)
        descriptors['Chi0'] = rdMolDescriptors.CalcChi0n(mol)
        descriptors['Chi0n'] = rdMolDescriptors.CalcChi0n(mol)
        descriptors['Chi0v'] = rdMolDescriptors.CalcChi0v(mol)
        descriptors['Chi1'] = rdMolDescriptors.CalcChi1n(mol)
        descriptors['Chi1n'] = rdMolDescriptors.CalcChi1n(mol)
        descriptors['Chi1v'] = rdMolDescriptors.CalcChi1v(mol)
        descriptors['Chi2n'] = rdMolDescriptors.CalcChi2n(mol)
        descriptors['Chi2v'] = rdMolDescriptors.CalcChi2v(mol)
        descriptors['Chi3n'] = rdMolDescriptors.CalcChi3n(mol)
        descriptors['Chi3v'] = rdMolDescriptors.CalcChi3v(mol)
        descriptors['Chi4n'] = rdMolDescriptors.CalcChi4n(mol)
        descriptors['Chi4v'] = rdMolDescriptors.CalcChi4v(mol)
        
        # Molecular properties
        descriptors['ExactMolWt'] = Descriptors.ExactMolWt(mol)
        descriptors['FractionCSP3'] = rdMolDescriptors.CalcFractionCsp3(mol)
        descriptors['HallKierAlpha'] = rdMolDescriptors.CalcHallKierAlpha(mol)
        descriptors['HeavyAtomCount'] = rdMolDescriptors.CalcNumHeavyAtoms(mol)
        descriptors['HeavyAtomMolWt'] = Descriptors.HeavyAtomMolWt(mol)
        descriptors['Kappa1'] = rdMolDescriptors.CalcKappa1(mol)
        descriptors['Kappa2'] = rdMolDescriptors.CalcKappa2(mol)
        descriptors['Kappa3'] = rdMolDescriptors.CalcKappa3(mol)
        descriptors['LabuteASA'] = rdMolDescriptors.CalcLabuteASA(mol)
        descriptors['MolLogP'] = Descriptors.MolLogP(mol)
        descriptors['MolMR'] = Descriptors.MolMR(mol)
        descriptors['MolWt'] = Descriptors.MolWt(mol)
        descriptors['NHOHCount'] = Descriptors.NHOHCount(mol)
        descriptors['NOCount'] = Descriptors.NOCount(mol)
        descriptors['NumAliphaticCarbocycles'] = rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)
        descriptors['NumAliphaticHeterocycles'] = rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)
        descriptors['NumAliphaticRings'] = rdMolDescriptors.CalcNumAliphaticRings(mol)
        descriptors['NumAromaticCarbocycles'] = rdMolDescriptors.CalcNumAromaticCarbocycles(mol)
        descriptors['NumAromaticHeterocycles'] = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
        descriptors['NumAromaticRings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
        descriptors['NumHAcceptors'] = rdMolDescriptors.CalcNumHBA(mol)
        descriptors['NumHDonors'] = rdMolDescriptors.CalcNumHBD(mol)
        descriptors['NumHeteroatoms'] = rdMolDescriptors.CalcNumHeteroatoms(mol)
        descriptors['NumRotatableBonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol)
        descriptors['NumSaturatedCarbocycles'] = rdMolDescriptors.CalcNumSaturatedCarbocycles(mol)
        descriptors['NumSaturatedHeterocycles'] = rdMolDescriptors.CalcNumSaturatedHeterocycles(mol)
        descriptors['NumSaturatedRings'] = rdMolDescriptors.CalcNumSaturatedRings(mol)
        descriptors['NumValenceElectrons'] = Descriptors.NumValenceElectrons(mol)
        descriptors['RingCount'] = rdMolDescriptors.CalcNumRings(mol)
        descriptors['TPSA'] = rdMolDescriptors.CalcTPSA(mol)
        
        # Partial charges
        try:
            AllChem.ComputeGasteigerCharges(mol)
            charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms() 
                      if not np.isnan(float(atom.GetProp('_GasteigerCharge')))]
            if charges:
                descriptors['MaxAbsPartialCharge'] = max([abs(c) for c in charges])
                descriptors['MaxPartialCharge'] = max(charges)
                descriptors['MinAbsPartialCharge'] = min([abs(c) for c in charges])
                descriptors['MinPartialCharge'] = min(charges)
            else:
                for key in ['MaxAbsPartialCharge', 'MaxPartialCharge', 'MinAbsPartialCharge', 'MinPartialCharge']:
                    descriptors[key] = 0.0
        except:
            for key in ['MaxAbsPartialCharge', 'MaxPartialCharge', 'MinAbsPartialCharge', 'MinPartialCharge']:
                descriptors[key] = 0.0
        
        return descriptors, None
    except Exception as e:
        return None, f"Error: {str(e)}"

def mol_to_image(smiles, size=(300, 300)):
    """Convert SMILES to molecule image"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Draw.MolToImage(mol, size=size)
    except:
        return None

def create_gauge_chart(probability):
    """Create a gauge chart for risk probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "DIA Risk Score", 'font': {'size': 24}},
        delta={'reference': 40, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': '#38ef7d'},
                {'range': [40, 60], 'color': '#ffd200'},
                {'range': [60, 80], 'color': '#ff6b6b'},
                {'range': [80, 100], 'color': '#c0392b'}
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_probability_distribution(probability):
    """Create probability distribution visualization"""
    x = np.linspace(0, 1, 100)
    y = np.exp(-((x - probability) ** 2) / 0.01)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.5)'))
    fig.add_vline(x=0.4, line_dash="dash", line_color="green", annotation_text="Threshold (0.4)")
    fig.add_vline(x=probability, line_dash="solid", line_color="red", annotation_text=f"Prediction ({probability:.3f})")
    fig.update_layout(title='Prediction Probability Distribution', height=300, showlegend=False)
    return fig

# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def show_home_page():
    """Home page content"""
    st.markdown("## 🎯 Welcome to DIA Risk Predictor")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h2>🔬</h2><h3>Advanced AI</h3><p>Stacking Ensemble Model</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h2>⚡</h2><h3>Fast Results</h3><p>Predictions in Seconds</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h2>🎯</h2><h3>High Accuracy</h3><p>93.9% ROC-AUC</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 🧪 Sample Compounds to Try")
    
    samples = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Penicillin": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"
    }
    
    cols = st.columns(4)
    for idx, (name, smiles) in enumerate(samples.items()):
        with cols[idx]:
            st.markdown(f"**{name}**")
            st.code(smiles, language=None)

def display_prediction_results(name, smiles, prediction, risk_category, probability):
    """Display prediction results"""
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"### {name if name else 'Unknown Compound'}")
        st.code(smiles)
    with col2:
        img = mol_to_image(smiles, size=(250, 250))
        if img:
            st.image(img)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown("### Classification")
        if prediction == 1:
            st.error("**⚠️ DIA POSITIVE**")
        else:
            st.success("**✅ DIA NEGATIVE**")
    
    with col2:
        st.markdown("### Risk Category")
        risk_class = f"risk-{risk_category.lower().replace(' ', '-')}"
        st.markdown(f'<div class="{risk_class}">{risk_category} Risk</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown("### Probability")
        st.metric("DIA Risk Score", f"{probability*100:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_gauge_chart(probability), use_container_width=True)
    with col2:
        st.plotly_chart(create_probability_distribution(probability), use_container_width=True)
    
    st.markdown("---")
    
    # Download report
    report_data = {
        "Compound Name": name if name else "Unknown",
        "SMILES": smiles,
        "Prediction": "DIA Positive" if prediction == 1 else "DIA Negative",
        "Risk Category": risk_category,
        "Probability": f"{probability*100:.2f}%",
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    report_df = pd.DataFrame([report_data])
    csv = report_df.to_csv(index=False)
    st.download_button(
        label="📄 Download Report (CSV)",
        data=csv,
        file_name=f"DIA_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def show_prediction_page(model):
    """Prediction page"""
    st.markdown("## 🔮 DIA Risk Prediction")
    
    st.markdown("### Enter Compound Information")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        compound_name = st.text_input("Compound Name (Optional):", placeholder="e.g., Aspirin")
        smiles = st.text_input("SMILES String:", placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O")
    
    with col2:
        st.markdown("### Molecule Preview")
        if smiles:
            img = mol_to_image(smiles)
            if img:
                st.image(img, caption="Molecular Structure")
            else:
                st.error("Invalid SMILES")
    
    if st.button("🔮 Predict DIA Risk", type="primary"):
        if not smiles:
            st.error("Please enter a SMILES string")
            return
        
        with st.spinner("Calculating..."):
            descriptors, error = calculate_all_rdkit_descriptors(smiles)
        
        if error:
            st.error(f"Error: {error}")
            return
        
        try:
            X_input = pd.DataFrame([descriptors])
            prediction, risk_category, probability = model.predict_with_risk(X_input)
            st.success("✅ Prediction Complete!")
            display_prediction_results(compound_name, smiles, prediction[0], risk_category[0], probability[0])
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

def add_footer():
    """Add footer"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>DIA Risk Predictor v1.0</strong></p>
        <p>Powered by Stacking Ensemble ML | ROC-AUC: 93.9% | Recall: 90.0%</p>
        <p>⚠️ <em>For research purposes only</em></p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.markdown('<p class="main-header">💊 Drug-Induced Autoimmunity Risk Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered DIA Risk Assessment System</p>', unsafe_allow_html=True)
    
    model = load_model()
    
    if model is None:
        st.error("Cannot load model")
        st.stop()
    
    with st.sidebar:
        st.image("https://img.icons8.com/cotton/200/000000/pill.png", width=150)
        st.markdown("### 📋 Navigation")
        page = st.radio("", ["🏠 Home", "🔮 Prediction"])
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("Stacking Ensemble ML model\n93.9% ROC-AUC\n90% Recall at threshold 0.4")
        
        st.markdown("---")
        st.markdown("### 📈 Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "85.8%")
            st.metric("Recall", "90.0%")
        with col2:
            st.metric("Precision", "65.9%")
            st.metric("F1-Score", "76.1%")
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Prediction":
        show_prediction_page(model)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
    add_footer()