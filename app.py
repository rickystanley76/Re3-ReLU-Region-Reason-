"""
Re3 Interactive Application - Streamlit UI
Interactive interface for ReLU Region Reason analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import io
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from openai import OpenAI

from re3_core import (
    extract_model_parameters,
    compute_region_affine_1layer,
    compute_region_affine_2layer,
    compute_region_affine_3layer,
    explain_region,
    analyze_regions
)

# Load environment variables
load_dotenv()

# Initialize OpenAI client for OpenRouter
def get_openrouter_client():
    """Initialize OpenRouter client with API key from environment."""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        return None
    
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

def get_llm_explanation(prompt: str, context: str = "") -> str:
    """
    Get explanation from GPT-4.1 mini via OpenRouter.
    
    Args:
        prompt: The main prompt/question
        context: Additional context information
        
    Returns:
        Explanation text from LLM
    """
    client = get_openrouter_client()
    if client is None:
        return "⚠️ OpenRouter API key not configured. Please set OPENROUTER_API_KEY in your .env file."
    
    try:
        model = os.getenv('OPENROUTER_MODEL', 'openai/gpt-4.1-mini')
        
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in neural network interpretability and explainable AI. Provide detailed, clear, concise, and technical explanations about ReLU neural networks and their interpretability."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error getting LLM explanation: {str(e)}"

# Page configuration
st.set_page_config(
    page_title="Re3: ReLU Region Reason",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_params' not in st.session_state:
    st.session_state.model_params = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None


def load_model(uploaded_file):
    """Load a PyTorch model from file.
    
    Note: Uses weights_only=False to support loading full model architectures.
    Only load model files from trusted sources.
    """
    try:
        if uploaded_file.name.endswith('.pth') or uploaded_file.name.endswith('.pt'):
            buffer = io.BytesIO(uploaded_file.read())
            # PyTorch 2.6+ defaults to weights_only=True for security.
            # Set weights_only=False to allow loading full models (nn.Sequential, etc.)
            loaded = torch.load(buffer, map_location='cpu', weights_only=False)
            
            # Check if it's a full model or just state dict
            if isinstance(loaded, nn.Module):
                # Full model saved
                return loaded, None
            else:
                # State dict - need to create model architecture
                st.warning("⚠️ State dict detected. Please create model architecture in sidebar or use 'Create New Model' option.")
                return None, loaded
        elif uploaded_file.name.endswith('.pkl'):
            # Load pickled model
            buffer = io.BytesIO(uploaded_file.read())
            model = pickle.load(buffer)
            if isinstance(model, nn.Module):
                return model, None
            else:
                st.error("Pickled file does not contain a PyTorch model.")
                return None, None
        else:
            st.error("Unsupported file format. Please upload .pth, .pt, or .pkl file.")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def create_model_from_config(input_size, hidden_sizes, output_size, num_layers):
    """Create a model based on configuration."""
    layers = []
    
    # Input to first hidden
    layers.append(nn.Linear(input_size, hidden_sizes[0]))
    layers.append(nn.ReLU())
    
    # Additional hidden layers
    for i in range(1, num_layers):
        layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        layers.append(nn.ReLU())
    
    # Output layer
    layers.append(nn.Linear(hidden_sizes[-1], output_size))
    
    return nn.Sequential(*layers)


def load_dataset(file_path, file_type='csv'):
    """Load dataset from file.

    Notes:
    - Many of the provided example datasets are **semicolon-delimited** (`;`)
      even though they use a `.csv` extension.
    - To make uploads robust, we let pandas **auto-detect** the delimiter
      using `sep=None` with the Python engine.
    """
    try:
        if file_type == 'csv':
            # Robust delimiter detection (handles ',' and ';')
            df = pd.read_csv(file_path, sep=None, engine='python')
        elif file_type == 'txt':
            # Try whitespace separator (handles tabs and spaces, including inconsistent fields)
            try:
                df = pd.read_csv(file_path, sep='\s+', header=None, engine='python')
            except:
                # Fallback to tab separator with error handling
                try:
                    df = pd.read_csv(file_path, sep='\t', header=None, on_bad_lines='skip')
                except:
                    df = pd.read_csv(file_path, sep=' ', header=None, on_bad_lines='skip')
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None


def prepare_data(df, target_column=None):
    """Prepare data for training/testing."""
    if target_column is None:
        # Assume last column is target
        target_column = df.columns[-1]
    
    # Get feature columns (exclude target)
    feature_df = df.drop(columns=[target_column])
    
    # Automatically drop non-numeric columns (e.g., timestamp, ID columns)
    # This handles cases like accelerometer dataset with timestamp column
    numeric_columns = []
    dropped_columns = []
    
    for col in feature_df.columns:
        # Try to convert to numeric, if it fails, it's non-numeric
        try:
            pd.to_numeric(feature_df[col], errors='raise')
            numeric_columns.append(col)
        except (ValueError, TypeError):
            dropped_columns.append(col)
    
    if dropped_columns:
        st.info(f"ℹ️ Dropped non-numeric columns: {', '.join(dropped_columns)}")
    
    # Use only numeric columns as features
    if len(numeric_columns) == 0:
        st.error("❌ No numeric feature columns found. Please check your dataset.")
        return None
    
    X = feature_df[numeric_columns].values
    y = df[target_column].values
    
    # Encode labels if needed
    le = LabelEncoder()
    if y.dtype == object:
        y = le.fit_transform(y)
        class_names = le.classes_.tolist()
    else:
        # Convert to integer to ensure proper indexing
        y = y.astype(int)
        # Get unique classes and ensure class_names covers all indices from 0 to max
        unique_classes = np.unique(y)
        max_class = int(unique_classes.max())
        # Create class_names list that covers all indices from 0 to max_class
        class_names = [str(i) if i in unique_classes else f'Class_{i}' for i in range(max_class + 1)]
    
    # Get feature names (only numeric columns)
    feature_names = numeric_columns
    
    # Use all data for analysis (no train/test split needed for interpretation)
    # Scale features using all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return {
        'X_train': X_scaled,  # Keep for compatibility, but contains all data
        'X_test': X_scaled,   # Use all data for analysis
        'y_train': y,         # Keep for compatibility, but contains all data
        'y_test': y,          # Use all data for analysis
        'feature_names': feature_names,
        'class_names': class_names,
        'scaler': scaler,
        'label_encoder': le
    }


# Main UI
st.markdown('<h1 class="main-header">🧠 Re3: ReLU Region Reason</h1>', unsafe_allow_html=True)
st.markdown("### Interactive Neural Network Interpretability Tool")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model loading section
    st.subheader("1. Load Model")
    uploaded_model = st.file_uploader(
        "Upload PyTorch Model (.pth, .pt, .pkl)",
        type=['pth', 'pt', 'pkl'],
        key="model_upload"
    )
    
    if uploaded_model:
        model, state_dict = load_model(uploaded_model)
        if model:
            st.session_state.model = model
            st.session_state.model_params = extract_model_parameters(model)
            st.success("✅ Model loaded successfully!")
        elif state_dict:
            st.warning("⚠️ State dict detected. Please upload a full model file (.pth with complete architecture).")
            st.session_state.model_state_dict = state_dict
    
    # Dataset loading section
    st.subheader("2. Load Dataset")
    uploaded_dataset = st.file_uploader(
        "Upload Dataset (.csv, .txt)",
        type=['csv', 'txt'],
        key="dataset_upload"
    )
    
    if uploaded_dataset:
        file_type = 'csv' if uploaded_dataset.name.endswith('.csv') else 'txt'
        df = load_dataset(uploaded_dataset, file_type)
        
        if df is not None:
            st.session_state.dataset = df
            st.dataframe(df.head(), width='stretch')
            
            # Target column selection
            target_col = st.selectbox(
                "Select Target Column",
                df.columns.tolist(),
                index=len(df.columns)-1
            )
            
            if st.button("Prepare Dataset"):
                data_prep = prepare_data(df, target_col)
                if data_prep is not None:
                    st.session_state.X_test = data_prep['X_test']
                    st.session_state.y_test = data_prep['y_test']
                    st.session_state.feature_names = data_prep['feature_names']
                    st.session_state.class_names = data_prep['class_names']
                    st.session_state.scaler = data_prep['scaler']
                    st.session_state.label_encoder = data_prep['label_encoder']
                    st.success("✅ Dataset prepared!")
                else:
                    st.error("❌ Failed to prepare dataset. Please check your data.")


# Main content area
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔍 Single Sample Analysis", "📈 Region Analysis", "📋 Model Info", "ℹ️ About"])

with tab1:
    st.header("Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### What is Re3?
        
        **Re3 (ReLU Region Reason)** is a method for interpreting ReLU neural networks 
        through piecewise-affine mapping. It provides:
        
        - ✅ **Exact explanations** for predictions
        - ✅ **Region identification** (activation patterns)
        - ✅ **Feature contributions** at logit and probability levels
        - ✅ **Neuron-level analysis**
        
        ### How to Use
        
        1. **Load or create a model** in the sidebar
        2. **Load a dataset** (upload or use built-in)
        3. **Analyze samples** in the "Single Sample Analysis" tab
        4. **Explore regions** in the "Region Analysis" tab
        """)
    
    with col2:
        st.markdown("""
        ### Current Status
        
        Check the status of your loaded components:
        """)
        
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            model_status = "✅ Loaded" if st.session_state.model is not None else "❌ Not Loaded"
            st.metric("Model", model_status)
            
            dataset_status = "✅ Loaded" if st.session_state.X_test is not None else "❌ Not Loaded"
            st.metric("Dataset", dataset_status)
        
        with status_col2:
            if st.session_state.model_params:
                num_layers = st.session_state.model_params.get('num_layers', 'Unknown')
                st.metric("Hidden Layers", num_layers)
            
            if st.session_state.X_test is not None:
                n_samples = len(st.session_state.X_test)
                st.metric("Test Samples", n_samples)


with tab2:
    st.header("Single Sample Analysis")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please load a model first in the sidebar.")
    elif st.session_state.X_test is None:
        st.warning("⚠️ Please load a dataset first in the sidebar.")
    else:
        # Extract model parameters
        if st.session_state.model_params is None:
            st.session_state.model_params = extract_model_parameters(st.session_state.model)
        
        params = st.session_state.model_params
        num_layers = params.get('num_layers', 0)
        
        # Validate that we have the required parameters
        if num_layers == 0:
            st.error("⚠️ Could not determine number of layers. Model may not have any Linear layers.")
            st.stop()
        
        # Check if we have all required parameters for the layer count
        required_params = []
        if num_layers == 1:
            required_params = ['w0', 'b0', 'w1', 'b1']
        elif num_layers == 2:
            required_params = ['w0', 'b0', 'w1', 'b1', 'w2', 'b2']
        elif num_layers == 3:
            required_params = ['w0', 'b0', 'w1', 'b1', 'w2', 'b2', 'w3', 'b3']
        
        missing_params = [p for p in required_params if p not in params]
        if missing_params:
            st.error(f"⚠️ Missing required parameters: {missing_params}")
            st.info(f"Available parameters: {list(params.keys())}")
            st.info(f"Model has {num_layers} hidden layer(s), which requires {len(required_params)//2} total Linear layers.")
            st.stop()
        
        # Sample selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            sample_idx = st.slider(
                "Select Sample Index",
                min_value=0,
                max_value=len(st.session_state.X_test) - 1,
                value=0,
                step=1
            )
        
        with col2:
            if st.button("Random Sample"):
                sample_idx = np.random.randint(0, len(st.session_state.X_test))
                st.rerun()
        
        # Get sample
        x = st.session_state.X_test[sample_idx]
        y_true = st.session_state.y_test[sample_idx]
        
        # Compute region-specific affine map
        # Note: compute functions use 1-indexed parameters (w1, w2, ...)
        # while extract_model_parameters uses 0-indexed (w0, w1, ...)
        if num_layers == 1:
            A_r, D_r, logits, pred, c1, region_id = compute_region_affine_1layer(
                x, params['w0'], params['b0'], params['w1'], params['b1']
            )
            neuron_contribs = {'Layer 1': c1}
        elif num_layers == 2:
            A_r, D_r, logits, pred, c1, c2, region_id = compute_region_affine_2layer(
                x, params['w0'], params['b0'], params['w1'], params['b1'],
                params['w2'], params['b2']
            )
            neuron_contribs = {'Layer 1': c1, 'Layer 2': c2}
        elif num_layers == 3:
            A_r, D_r, logits, pred, c1, c2, c3, region_id = compute_region_affine_3layer(
                x, params['w0'], params['b0'], params['w1'], params['b1'],
                params['w2'], params['b2'], params['w3'], params['b3']
            )
            neuron_contribs = {'Layer 1': c1, 'Layer 2': c2, 'Layer 3': c3}
        else:
            st.error(f"Unsupported number of layers: {num_layers}. Supported: 1-3 hidden layers.")
            st.stop()
        
        # Compute probabilities
        from scipy.special import softmax
        probs = softmax(logits)
        
        # Store in session state for AI explanation
        st.session_state.current_sample_data = {
            'x': x,
            'y_true': y_true,
            'pred': pred,
            'logits': logits,
            'probs': probs,
            'region_id': region_id,
            'neuron_contribs': neuron_contribs,
            'A_r': A_r,
            'D_r': D_r
        }
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Convert to int to ensure proper indexing (handles numpy float64)
            y_true_int = int(y_true) if isinstance(y_true, (np.floating, np.integer)) else y_true
            # Bounds check to prevent IndexError
            if 0 <= y_true_int < len(st.session_state.class_names):
                true_label = st.session_state.class_names[y_true_int]
            else:
                true_label = f"Class_{y_true_int}"
            st.metric("True Label", true_label)
        with col2:
            # Ensure pred is int (should already be from compute functions)
            pred_int = int(pred) if isinstance(pred, (np.floating, np.integer)) else pred
            # Bounds check to prevent IndexError
            if 0 <= pred_int < len(st.session_state.class_names):
                pred_label = st.session_state.class_names[pred_int]
            else:
                pred_label = f"Class_{pred_int}"
            st.metric("Predicted Label", pred_label)
        with col3:
            pred_int = int(pred) if isinstance(pred, (np.floating, np.integer)) else pred
            # Bounds check for probability array
            if 0 <= pred_int < len(probs):
                confidence = f"{probs[pred_int]:.2%}"
            else:
                confidence = "N/A"
            st.metric("Confidence", confidence)
        
        st.markdown("---")
        
        # Feature contributions
        st.subheader("Feature Contributions")
        df_exp = explain_region(
            x, A_r, D_r, logits, pred,
            st.session_state.feature_names,
            st.session_state.class_names
        )
        
        # Sort by absolute probability contribution
        df_exp_sorted = df_exp.sort_values('ProbContribution', key=abs, ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(df_exp_sorted, width='stretch')
        
        with col2:
            # Bar chart of contributions
            fig, ax = plt.subplots(figsize=(6, 8))
            colors = ['red' if x < 0 else 'green' for x in df_exp_sorted['ProbContribution']]
            ax.barh(df_exp_sorted['Feature'], df_exp_sorted['ProbContribution'], color=colors)
            ax.set_xlabel('Probability Contribution')
            ax.set_title('Feature Contributions')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Neuron contributions
        st.subheader("Neuron Contributions")
        
        for layer_name, contribs in neuron_contribs.items():
            st.markdown(f"**{layer_name}**")
            df_neurons = pd.DataFrame({
                'Neuron': [f'Neuron {i+1}' for i in range(len(contribs))],
                'Contribution': contribs
            }).sort_values('Contribution', key=abs, ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(df_neurons, width='stretch')
            
            with col2:
                fig, ax = plt.subplots(figsize=(6, max(4, len(contribs) * 0.3)))
                colors = ['red' if x < 0 else 'green' for x in df_neurons['Contribution']]
                ax.barh(df_neurons['Neuron'], df_neurons['Contribution'], color=colors)
                ax.set_xlabel('Contribution')
                ax.set_title(f'{layer_name} Neuron Contributions')
                ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig)
        
        # Region information
        st.subheader("Region Information")
        st.code(f"Region ID: {region_id}")
        
        # Class probabilities
        st.subheader("Class Probabilities")
        # Ensure class_names matches the number of output classes from model
        num_output_classes = len(probs)
        if len(st.session_state.class_names) < num_output_classes:
            # Extend class_names if model has more output classes
            extended_class_names = list(st.session_state.class_names)
            for i in range(len(st.session_state.class_names), num_output_classes):
                extended_class_names.append(f'Class_{i}')
            class_names_for_probs = extended_class_names
        elif len(st.session_state.class_names) > num_output_classes:
            # Truncate if class_names has more entries than model outputs
            class_names_for_probs = st.session_state.class_names[:num_output_classes]
        else:
            class_names_for_probs = st.session_state.class_names
        
        df_probs = pd.DataFrame({
            'Class': class_names_for_probs,
            'Probability': probs,
            'Logit': logits
        }).sort_values('Probability', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(df_probs, width='stretch')
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(df_probs['Class'], df_probs['Probability'])
            ax.set_ylabel('Probability')
            ax.set_title('Class Probabilities')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        # LLM Explanation Section
        st.markdown("---")
        st.subheader("🤖 AI Explanation")
        
        if st.button("Get AI Explanation", key="explain_sample"):
            try:
                with st.spinner("Generating explanation..."):
                    # Check if sample data exists
                    if 'current_sample_data' not in st.session_state:
                        st.error("⚠️ Please select a sample first to generate analysis data.")
                        st.stop()
                    
                    sample_data = st.session_state.current_sample_data
                    
                    # Recompute feature contributions if needed
                    df_exp = explain_region(
                        sample_data['x'], 
                        sample_data['A_r'], 
                        sample_data['D_r'], 
                        sample_data['logits'], 
                        sample_data['pred'],
                        st.session_state.feature_names,
                        st.session_state.class_names
                    )
                    df_exp_sorted = df_exp.sort_values('ProbContribution', key=abs, ascending=False)
                    
                    # Prepare context for LLM
                    top_features = df_exp_sorted.head(5)
                    top_features_str = "\n".join([
                        f"- {row['Feature']}: {row['ProbContribution']:.4f} (contribution to probability)"
                        for _, row in top_features.iterrows()
                    ])
                    
                    # Get neuron contributions
                    top_neurons = []
                    neuron_contribs = sample_data['neuron_contribs']
                    for layer_name, contribs in neuron_contribs.items():
                        if len(contribs) > 0:
                            df_neurons = pd.DataFrame({
                                'Neuron': [f'Neuron {i+1}' for i in range(len(contribs))],
                                'Contribution': contribs
                            }).sort_values('Contribution', key=abs, ascending=False)
                            top_neurons.append(f"{layer_name}: {df_neurons.iloc[0]['Neuron']} (contribution: {df_neurons.iloc[0]['Contribution']:.4f})")
                    
                    # Get labels
                    y_true_int = int(sample_data['y_true']) if isinstance(sample_data['y_true'], (np.floating, np.integer)) else sample_data['y_true']
                    pred_int = int(sample_data['pred']) if isinstance(sample_data['pred'], (np.floating, np.integer)) else sample_data['pred']
                    
                    if 0 <= y_true_int < len(st.session_state.class_names):
                        true_label = st.session_state.class_names[y_true_int]
                    else:
                        true_label = f"Class_{y_true_int}"
                    
                    if 0 <= pred_int < len(st.session_state.class_names):
                        pred_label = st.session_state.class_names[pred_int]
                    else:
                        pred_label = f"Class_{pred_int}"
                    
                    # Get confidence
                    if 0 <= pred_int < len(sample_data['probs']):
                        confidence = f"{sample_data['probs'][pred_int]:.2%}"
                    else:
                        confidence = "N/A"
                    
                    # Get class names for probabilities
                    num_output_classes = len(sample_data['probs'])
                    if len(st.session_state.class_names) < num_output_classes:
                        extended_class_names = list(st.session_state.class_names)
                        for i in range(len(st.session_state.class_names), num_output_classes):
                            extended_class_names.append(f'Class_{i}')
                        class_names_for_probs = extended_class_names
                    elif len(st.session_state.class_names) > num_output_classes:
                        class_names_for_probs = st.session_state.class_names[:num_output_classes]
                    else:
                        class_names_for_probs = st.session_state.class_names
                    
                    # Build context
                    prob_str = ', '.join([f'{cls}: {prob:.2%}' for cls, prob in zip(class_names_for_probs, sample_data['probs'])])
                    
                    context = f"""
Sample Analysis Context:
- True Label: {true_label}
- Predicted Label: {pred_label}
- Confidence: {confidence}
- Region ID: {sample_data['region_id']}
- Top Contributing Features:
{top_features_str}
- Top Contributing Neurons:
{chr(10).join(top_neurons) if top_neurons else 'N/A'}
- Class Probabilities: {prob_str}
"""
                    
                    prompt = f"""
Explain this neural network prediction in simple terms:
1. Why did the model predict "{pred_label}" instead of "{true_label}"?
2. Which features were most important for this prediction and why?
3. What does the region ID "{sample_data['region_id']}" tell us about how the model processed this input?
4. How do the neuron contributions help us understand the model's decision-making process?
"""
                    
                    explanation = get_llm_explanation(prompt, context)
                    st.markdown(explanation)
            except KeyError as e:
                st.error(f"⚠️ Error: Missing data. Please ensure you've selected a sample and the analysis has completed. Error: {str(e)}")
            except Exception as e:
                st.error(f"⚠️ Error generating explanation: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


with tab3:
    st.header("Region Analysis")
    
    if st.session_state.model is None or st.session_state.X_test is None:
        st.warning("⚠️ Please load both model and dataset first.")
    else:
        if st.button("Analyze All Regions"):
            with st.spinner("Analyzing regions..."):
                # Extract model parameters if not already done
                if st.session_state.model_params is None:
                    st.session_state.model_params = extract_model_parameters(st.session_state.model)
                
                params = st.session_state.model_params
                num_layers = params.get('num_layers', 0)
                
                if num_layers == 0 or num_layers > 3:
                    st.error(f"⚠️ Unsupported number of layers: {num_layers}. Supported: 1-3 hidden layers.")
                    st.stop()
                
                # Get predictions for all test samples
                X_test_tensor = torch.tensor(st.session_state.X_test, dtype=torch.float32)
                st.session_state.model.eval()
                with torch.no_grad():
                    outputs = st.session_state.model(X_test_tensor)
                    y_pred = outputs.argmax(dim=1).numpy()
                
                # Analyze regions
                results = analyze_regions(
                    st.session_state.X_test,
                    st.session_state.y_test,
                    y_pred,
                    params,
                    st.session_state.feature_names,
                    st.session_state.class_names,
                    num_layers
                )
                
                st.session_state.region_results = results
        
        if 'region_results' in st.session_state:
            results = st.session_state.region_results
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Unique Regions", results['n_unique_regions'])
            with col2:
                avg_purity = results['region_stats']['purity'].mean()
                st.metric("Avg. Purity", f"{avg_purity:.2%}")
            with col3:
                avg_accuracy = results['region_stats']['accuracy'].mean()
                st.metric("Avg. Accuracy", f"{avg_accuracy:.2%}")
            with col4:
                total_samples = results['region_stats']['n_samples'].sum()
                st.metric("Total Samples", total_samples)
            
            st.markdown("---")
            
            # Region statistics table
            st.subheader("Region Statistics")
            st.dataframe(results['region_stats'], width='stretch')
            
            # Sankey diagram showing flow from regions to classes
            st.subheader("Region to Class Flow (Sankey Diagram)")
            
            # Prepare data for Sankey diagram
            region_stats_df = results['region_stats'].copy()
            
            # Get number of layers from model params
            params = st.session_state.model_params
            num_layers = params.get('num_layers', 1)
            
            # Parse region_id to extract layer information
            def parse_region_id(rid):
                """Parse region_id to extract L1 and L2 components."""
                if isinstance(rid, str):
                    # Check if it's a 2-layer format: "L1:11110000_L2:00110011"
                    if '_L2:' in rid:
                        parts = rid.split('_L2:')
                        l1_part = parts[0].replace('L1:', '')
                        l2_part = parts[1]
                        return {'l1': l1_part, 'l2': l2_part, 'full': rid}
                    # Check if it's a 1-layer format: "L1:11110000"
                    elif rid.startswith('L1:'):
                        l1_part = rid.replace('L1:', '')
                        return {'l1': l1_part, 'l2': None, 'full': rid}
                    # Check if it's a 3-layer format: "L1:..._L2:..._L3:..."
                    elif '_L3:' in rid:
                        parts = rid.split('_L3:')
                        l1_l2 = parts[0]
                        if '_L2:' in l1_l2:
                            l1_part = l1_l2.split('_L2:')[0].replace('L1:', '')
                            l2_part = l1_l2.split('_L2:')[1]
                            l3_part = parts[1]
                            return {'l1': l1_part, 'l2': l2_part, 'l3': l3_part, 'full': rid}
                    return {'l1': None, 'l2': None, 'full': rid}
                else:
                    return {'l1': None, 'l2': None, 'full': str(rid)}
            
            # Check if we have multi-layer region IDs
            sample_rid = region_stats_df['region_id'].iloc[0] if len(region_stats_df) > 0 else None
            parsed_sample = parse_region_id(sample_rid) if sample_rid else None
            has_l2 = parsed_sample and parsed_sample['l2'] is not None
            
            # For 2-layer models, create multi-layer Sankey diagram
            if num_layers == 2 and has_l2:
                # Parse all region IDs
                region_stats_df['parsed'] = region_stats_df['region_id'].apply(parse_region_id)
                
                # Extract unique L1 and L2 patterns
                l1_patterns = region_stats_df['parsed'].apply(lambda x: x['l1']).unique()
                l2_patterns = region_stats_df['parsed'].apply(lambda x: x['l2']).unique()
                unique_classes = region_stats_df['majority_class'].unique().tolist()
                unique_classes.sort()
                
                # Get feature names
                feature_names = st.session_state.feature_names if st.session_state.feature_names else [f'Feature_{i+1}' for i in range(len(l1_patterns))]
                
                # Create node labels for 4 columns: Features → L1 → L2 → Classes
                # Column 1: Features
                feature_nodes = feature_names[:len(feature_names)]  # Use actual feature names
                
                # Column 2: L1 regions (format as "L1 nX" where X is index)
                l1_nodes = [f"L1 n{i+1}" for i in range(len(l1_patterns))]
                l1_pattern_to_node = {pattern: f"L1 n{i+1}" for i, pattern in enumerate(sorted(l1_patterns))}
                
                # Column 3: L2 regions (format as "L2 nX" where X is index)
                l2_nodes = [f"L2 n{i+1}" for i in range(len(l2_patterns))]
                l2_pattern_to_node = {pattern: f"L2 n{i+1}" for i, pattern in enumerate(sorted(l2_patterns))}
                
                # Column 4: Classes
                class_nodes = unique_classes
                
                # Combine all nodes
                all_nodes = feature_nodes + l1_nodes + l2_nodes + class_nodes
                node_indices = {node: idx for idx, node in enumerate(all_nodes)}
                
                # Calculate feature contributions to L1 regions
                # For each L1 pattern, compute which features contribute most
                # We'll use a heuristic: features with higher variance in L1 activation patterns
                l1_feature_flows = {}  # {(feature, l1_node): count}
                l1_l2_flows = {}  # {(l1_node, l2_node): count}
                l2_class_flows = {}  # {(l2_node, class): count}
                
                # Aggregate flows from region_stats
                for idx, row in region_stats_df.iterrows():
                    parsed = row['parsed']
                    l1_pattern = parsed['l1']
                    l2_pattern = parsed['l2']
                    class_name = str(row['majority_class'])
                    n_samples = int(row['n_samples'])
                    
                    if l1_pattern and l2_pattern:
                        l1_node = l1_pattern_to_node[l1_pattern]
                        l2_node = l2_pattern_to_node[l2_pattern]
                        
                        # L1 → L2 flow
                        key = (l1_node, l2_node)
                        l1_l2_flows[key] = l1_l2_flows.get(key, 0) + n_samples
                        
                        # L2 → Class flow
                        key = (l2_node, class_name)
                        l2_class_flows[key] = l2_class_flows.get(key, 0) + n_samples
                
                # For Features → L1, compute actual feature contributions
                # Use the regions dictionary to get sample-level data
                regions_dict = results.get('regions', {})
                
                # Aggregate feature contributions per L1 pattern
                l1_feature_contributions = {}  # {l1_pattern: {feature_idx: total_contribution}}
                
                if regions_dict and st.session_state.X_test is not None:
                    for region_id, region_data in regions_dict.items():
                        parsed = parse_region_id(region_id)
                        if parsed['l1'] and parsed['l2']:
                            l1_pattern = parsed['l1']
                            samples = region_data.get('samples', [])
                            
                            if l1_pattern not in l1_feature_contributions:
                                l1_feature_contributions[l1_pattern] = {i: 0.0 for i in range(len(feature_nodes))}
                            
                            # For each sample in this region, compute feature contributions
                            # Use absolute feature values as a proxy for contribution
                            for sample_idx in samples:
                                if sample_idx < len(st.session_state.X_test):
                                    x = st.session_state.X_test[sample_idx]
                                    # Use absolute values as contribution proxy
                                    for feat_idx in range(min(len(feature_nodes), len(x))):
                                        l1_feature_contributions[l1_pattern][feat_idx] += abs(x[feat_idx])
                
                # Normalize and create flows
                for l1_pattern, l1_node in l1_pattern_to_node.items():
                    # Count samples going through this L1 node
                    l1_total = sum(v for (l1, l2), v in l1_l2_flows.items() if l1 == l1_node)
                    
                    if l1_total > 0:
                        if l1_pattern in l1_feature_contributions and l1_feature_contributions[l1_pattern]:
                            # Use actual feature contributions
                            contributions = l1_feature_contributions[l1_pattern]
                            total_contrib = sum(contributions.values())
                            
                            if total_contrib > 0:
                                # Distribute based on actual feature contributions
                                for feat_idx, feature in enumerate(feature_nodes):
                                    if feat_idx < len(contributions):
                                        contrib_ratio = contributions[feat_idx] / total_contrib
                                        flow_value = int(l1_total * contrib_ratio)
                                        if flow_value > 0:
                                            key = (feature, l1_node)
                                            l1_feature_flows[key] = l1_feature_flows.get(key, 0) + flow_value
                        else:
                            # Fallback: distribute evenly across features
                            flow_per_feature = l1_total // len(feature_nodes)
                            remainder = l1_total % len(feature_nodes)
                            for feat_idx, feature in enumerate(feature_nodes):
                                flow_value = flow_per_feature + (1 if feat_idx < remainder else 0)
                                if flow_value > 0:
                                    key = (feature, l1_node)
                                    l1_feature_flows[key] = l1_feature_flows.get(key, 0) + flow_value
                
                # Build source, target, value lists
                source = []
                target = []
                value = []
                
                # Features → L1 flows
                for (feat, l1), val in l1_feature_flows.items():
                    if feat in node_indices and l1 in node_indices and val > 0:
                        source.append(node_indices[feat])
                        target.append(node_indices[l1])
                        value.append(val)
                
                # L1 → L2 flows
                for (l1, l2), val in l1_l2_flows.items():
                    if l1 in node_indices and l2 in node_indices and val > 0:
                        source.append(node_indices[l1])
                        target.append(node_indices[l2])
                        value.append(val)
                
                # L2 → Class flows
                for (l2, cls), val in l2_class_flows.items():
                    if l2 in node_indices and cls in node_indices and val > 0:
                        source.append(node_indices[l2])
                        target.append(node_indices[cls])
                        value.append(val)
                
                if len(source) == 0:
                    st.warning("⚠️ No valid flow data available for multi-layer Sankey diagram.")
                else:
                    # Color palette
                    feature_colors = [
                        "rgba(31, 119, 180, 0.8)",    # Blue
                        "rgba(255, 127, 14, 0.8)",     # Orange
                        "rgba(44, 160, 44, 0.8)",      # Green
                        "rgba(214, 39, 40, 0.8)",      # Red
                        "rgba(148, 103, 189, 0.8)",     # Purple
                        "rgba(140, 86, 75, 0.8)",      # Brown
                    ]
                    
                    l1_colors = ["rgba(174, 199, 232, 0.8)"] * len(l1_nodes)  # Light blue
                    l2_colors = ["rgba(255, 187, 120, 0.8)"] * len(l2_nodes)  # Light orange
                    
                    class_colors_palette = [
                        "rgba(227, 119, 194, 0.8)",    # Pink
                        "rgba(44, 160, 44, 0.8)",      # Green
                        "rgba(148, 103, 189, 0.8)",     # Purple
                        "rgba(255, 127, 14, 0.8)",     # Orange
                        "rgba(23, 190, 207, 0.8)",     # Cyan
                    ]
                    
                    class_colors = [class_colors_palette[i % len(class_colors_palette)] for i in range(len(class_nodes))]
                    
                    node_colors = (
                        [feature_colors[i % len(feature_colors)] for i in range(len(feature_nodes))] +
                        l1_colors +
                        l2_colors +
                        class_colors
                    )
                    
                    # Create Sankey diagram
                    fig = go.Figure(data=[go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color="black", width=0.5),
                            label=all_nodes,
                            color=node_colors
                        ),
                        link=dict(
                            source=source,
                            target=target,
                            value=value,
                            color="rgba(128, 128, 128, 0.4)"
                        )
                    )])
                    
                    height = max(500, min(1200, len(all_nodes) * 25))
                    fig.update_layout(
                        title_text="Multi-Layer Flow: Features → L1 Regions → L2 Regions → Classes",
                        font_size=10,
                        height=height,
                        width=1400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"💡 **Legend**: Features (colored) → L1 Regions (light blue) → L2 Regions (light orange) → Classes (colored). Flow width indicates sample count.")
            
            else:
                # Single-layer or simple 2-column diagram
                def format_region_label(rid):
                    """Format region ID as a readable label."""
                    if isinstance(rid, str):
                        if len(rid) > 30:
                            return f"R{hash(rid) % 10000}"
                        return rid
                    else:
                        try:
                            return f"Region {int(rid)}"
                        except (ValueError, TypeError):
                            return str(rid)
                
                region_labels = [format_region_label(rid) for rid in region_stats_df['region_id']]
                unique_classes = region_stats_df['majority_class'].unique().tolist()
                unique_classes.sort()
                
                all_nodes = region_labels + unique_classes
                node_indices = {node: idx for idx, node in enumerate(all_nodes)}
                
                source = []
                target = []
                value = []
                
                for idx, row in region_stats_df.iterrows():
                    region_node = format_region_label(row['region_id'])
                    class_node = str(row['majority_class'])
                    n_samples = int(row['n_samples'])
                    
                    if n_samples > 0 and region_node in node_indices and class_node in node_indices:
                        source.append(node_indices[region_node])
                        target.append(node_indices[class_node])
                        value.append(n_samples)
                
                if len(source) == 0:
                    st.warning("⚠️ No valid flow data available for Sankey diagram.")
                else:
                    class_colors = [
                        "rgba(31, 119, 180, 0.8)", "rgba(255, 127, 14, 0.8)", "rgba(44, 160, 44, 0.8)",
                        "rgba(214, 39, 40, 0.8)", "rgba(148, 103, 189, 0.8)", "rgba(140, 86, 75, 0.8)",
                        "rgba(227, 119, 194, 0.8)", "rgba(127, 127, 127, 0.8)", "rgba(188, 189, 34, 0.8)",
                        "rgba(23, 190, 207, 0.8)",
                    ]
                    
                    region_color = "rgba(174, 199, 232, 0.8)"
                    
                    node_colors = []
                    for node in all_nodes:
                        if node in unique_classes:
                            class_idx = unique_classes.index(node)
                            node_colors.append(class_colors[class_idx % len(class_colors)])
                        else:
                            node_colors.append(region_color)
                    
                    link_colors = []
                    for tgt_idx in target:
                        if tgt_idx >= len(region_labels):
                            class_node = all_nodes[tgt_idx]
                            if class_node in unique_classes:
                                class_idx = unique_classes.index(class_node)
                                link_colors.append(class_colors[class_idx % len(class_colors)].replace("0.8", "0.3"))
                            else:
                                link_colors.append("rgba(128, 128, 128, 0.3)")
                        else:
                            link_colors.append("rgba(128, 128, 128, 0.3)")
                    
                    fig = go.Figure(data=[go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color="black", width=0.5),
                            label=all_nodes,
                            color=node_colors
                        ),
                        link=dict(
                            source=source,
                            target=target,
                            value=value,
                            color=link_colors
                        )
                    )])
                    
                    height = max(400, min(1000, len(all_nodes) * 30))
                    fig.update_layout(
                        title_text="Flow from Regions to Majority Classes",
                        font_size=11,
                        height=height,
                        width=1200
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"💡 **Legend**: Left nodes (light blue) represent regions. Right nodes (colored) represent classes. Flow width indicates number of samples.")
            
            # LLM Explanation Section
            st.markdown("---")
            st.subheader("🤖 AI Explanation")
            
            if st.button("Get AI Explanation", key="explain_regions"):
                try:
                    with st.spinner("Generating explanation..."):
                        # Ensure results exist
                        if 'results' not in locals() and 'results' not in globals():
                            if 'region_results' in st.session_state:
                                results = st.session_state.region_results
                            else:
                                st.error("⚠️ Please run 'Analyze All Regions' first.")
                                st.stop()
                        
                        # Prepare context for LLM
                        region_stats_df = results['region_stats']
                        top_regions = region_stats_df.head(5)
                        top_regions_str = "\n".join([
                            f"- {row['region_id']}: {row['n_samples']} samples, purity: {row['purity']:.2%}, accuracy: {row['accuracy']:.2%}, majority class: {row['majority_class']}"
                            for _, row in top_regions.iterrows()
                        ])
                        
                        # Get class distribution from Sankey diagram data
                        unique_classes = region_stats_df['majority_class'].unique().tolist()
                        class_distribution = region_stats_df.groupby('majority_class')['n_samples'].sum().to_dict()
                        class_dist_str = "\n".join([
                            f"- {cls}: {count} total samples across regions"
                            for cls, count in sorted(class_distribution.items())
                        ])
                        
                        # Calculate metrics safely
                        avg_purity_val = results['region_stats']['purity'].mean()
                        avg_accuracy_val = results['region_stats']['accuracy'].mean()
                        total_samples_val = results['region_stats']['n_samples'].sum()
                        
                        # Get region size statistics
                        region_sizes = region_stats_df['n_samples'].values
                        min_size = int(region_sizes.min())
                        max_size = int(region_sizes.max())
                        median_size = int(np.median(region_sizes))
                        
                        context = f"""
Region Analysis Context:
- Total Unique Regions: {results['n_unique_regions']}
- Average Purity: {avg_purity_val:.2%} (proportion of majority class in each region)
- Average Accuracy: {avg_accuracy_val:.2%} (prediction accuracy within each region)
- Total Samples Analyzed: {total_samples_val}
- Region Size Range: {min_size} to {max_size} samples (median: {median_size})
- Number of Classes: {len(unique_classes)}
- Class Distribution:
{class_dist_str}
- Top 5 Regions by Sample Size:
{top_regions_str}
"""
                        
                        prompt = f"""
Analyze these region analysis results based on what's shown in the Region Analysis tab:

1. **Summary Metrics**: What do the key metrics (Unique Regions: {results['n_unique_regions']}, Avg. Purity: {avg_purity_val:.2%}, Avg. Accuracy: {avg_accuracy_val:.2%}) tell us about the model's behavior and decision-making structure?

2. **Region Statistics Table**: Based on the region statistics showing region_id, n_samples, purity, majority_class, and accuracy, what patterns can you identify? Are there regions with very high or very low purity/accuracy that might indicate model strengths or weaknesses?

3. **Sankey Diagram (Region-to-Class Flow)**: The Sankey diagram shows how samples flow from regions (left, light blue nodes) to their majority classes (right, colored nodes). What does this visualization reveal about:
   - How well regions map to specific classes?
   - Whether there's a clear separation between classes in the region space?
   - The distribution of samples across regions and classes?

4. **Model Interpretability**: How does this region analysis help us understand the model's internal mechanism? What insights can we gain about how the ReLU network partitions the input space into distinct linear regions?

5. **Potential Issues**: Are there any concerning patterns (e.g., regions with low purity, imbalanced region sizes, or regions mapping to wrong classes) that might indicate model problems or areas for improvement?
"""
                        
                        explanation = get_llm_explanation(prompt, context)
                        st.markdown(explanation)
                except KeyError as e:
                    st.error(f"⚠️ Error: Missing data in results. Please run 'Analyze All Regions' first. Error: {str(e)}")
                except Exception as e:
                    st.error(f"⚠️ Error generating explanation: {str(e)}")


with tab4:
    st.header("Model Information")
    
    if st.session_state.model is None:
        st.warning("⚠️ No model loaded.")
    else:
        st.subheader("Model Architecture")
        st.code(str(st.session_state.model))
        
        if st.session_state.model_params:
            st.subheader("Model Parameters")
            
            params = st.session_state.model_params
            num_layers = params['num_layers']
            
            for i in range(num_layers + 1):
                w_key = f'w{i}'
                b_key = f'b{i}'
                
                if w_key in params:
                    st.markdown(f"**Layer {i}**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"Weights shape: {params[w_key].shape}")
                        st.dataframe(pd.DataFrame(params[w_key]).round(4), width='stretch')
                    
                    with col2:
                        st.write(f"Biases shape: {params[b_key].shape}")
                        st.dataframe(pd.DataFrame(params[b_key]).round(4), width='stretch')
        
        if st.session_state.feature_names:
            st.subheader("Feature Information")
            st.write(f"Number of features: {len(st.session_state.feature_names)}")
            st.write("Feature names:", st.session_state.feature_names)
        
        if st.session_state.class_names:
            st.subheader("Class Information")
            st.write(f"Number of classes: {len(st.session_state.class_names)}")
            st.write("Class names:", st.session_state.class_names)
        
        # LLM Explanation Section
        st.markdown("---")
        st.subheader("🤖 AI Explanation")
        
        if st.button("Get AI Explanation", key="explain_model"):
            try:
                with st.spinner("Generating explanation..."):
                    # Prepare context for LLM
                    model_info = []
                    
                    if st.session_state.model is not None:
                        model_info.append(f"Model Architecture: {str(st.session_state.model)}")
                    else:
                        st.error("⚠️ No model loaded.")
                        st.stop()
                    
                    if st.session_state.model_params:
                        params = st.session_state.model_params
                        num_layers = params.get('num_layers', 0)
                        model_info.append(f"Number of Hidden Layers: {num_layers}")
                        
                        for i in range(num_layers + 1):
                            w_key = f'w{i}'
                            b_key = f'b{i}'
                            if w_key in params:
                                model_info.append(f"Layer {i}: Weights shape {params[w_key].shape}, Biases shape {params[b_key].shape}")
                    
                    if st.session_state.feature_names:
                        model_info.append(f"Number of Features: {len(st.session_state.feature_names)}")
                        feature_list = ', '.join(st.session_state.feature_names[:10])
                        if len(st.session_state.feature_names) > 10:
                            feature_list += '...'
                        model_info.append(f"Features: {feature_list}")
                    
                    if st.session_state.class_names:
                        model_info.append(f"Number of Classes: {len(st.session_state.class_names)}")
                        model_info.append(f"Classes: {', '.join(st.session_state.class_names)}")
                    
                    context = "\n".join(model_info)
                    
                    prompt = f"""
Explain this neural network model:
1. What does this model architecture tell us about its complexity and capacity?
2. How do the layer sizes and shapes affect the model's ability to learn?
3. What can we infer about the relationship between the number of features and the model structure?
4. How does this architecture relate to the Re3 interpretability method?
5. What are the implications of this model structure for interpretability and explainability?
"""
                    
                    explanation = get_llm_explanation(prompt, context)
                    st.markdown(explanation)
            except Exception as e:
                st.error(f"⚠️ Error generating explanation: {str(e)}")


with tab5:
    st.header("About Re3: ReLU Region Reason")
    
    st.markdown("""
    ## 📖 Project Description
    
    **Re3 (ReLU Region Reason):** Rectified linear unit (ReLU) based neural networks (NNs) is an interactive interpretability tool designed for understanding and explaining 
    the behavior of ReLU-based neural networks through piecewise-affine mapping. It is recognised for their remarkable accuracy. However, the decision-making processes of these 
    networks are often complex and difficult to understand. Our Re3 application provides 
    exact, mathematically-grounded explanations for neural network predictions by decomposing the network's 
    computation into distinct linear regions defined by ReLU activation patterns.
    
    Unlike black-box interpretability methods that provide approximate explanations, Re3 offers **exact 
    explanations** by identifying the specific linear region in which each input sample resides and computing 
    the precise affine transformation that maps inputs to outputs within that region.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 🎯 What This App Does
    
    Re3 enables users to:
    
    ### 1. **Single Sample Analysis**
    - **Exact Explanations**: Get precise feature-level contributions for individual predictions
    - **Neuron-Level Insights**: Understand which neurons in each layer contribute most to the prediction
    - **Region Identification**: Identify the specific activation pattern (region) for each sample
    - **Probability Decomposition**: See how each feature affects both logit(Logit(\(p\)) = log(\(p\) / (1-\(p\)))) and probability outputs
    
    ### 2. **Region Analysis**
    - **Region Discovery**: Automatically identify all unique linear regions in the input space
    - **Region Statistics**: Analyze region purity, accuracy, and size distribution
    - **Quality Metrics**: Evaluate how well regions correspond to class boundaries
    - **Visualization**: Explore region characteristics through interactive plots
    
    ### 3. **Model Understanding**
    - **Architecture Inspection**: View model weights, biases, and layer configurations
    - **Feature Analysis**: Understand how features are used across different regions
    - **Class Behavior**: Analyze how different classes are represented in the region space
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 👥 Target Audience
    
    This application is designed for:
    
    ### **Researchers & Academics**
    - Machine learning researchers studying neural network interpretability
    - Academics teaching or researching explainable AI (XAI)
    - Graduate students working on interpretability methods
    - Researchers investigating the geometric properties of neural networks
    
    ### **Data Scientists & ML Engineers**
    - Practitioners who need to explain model decisions to stakeholders
    - ML engineers debugging and validating neural network behavior
    - Data scientists requiring transparent model explanations for regulatory compliance
    - Professionals building trustworthy AI systems
    
    ### **Educators & Students**
    - Instructors teaching neural network interpretability
    - Students learning about explainable AI concepts
    - Anyone interested in understanding how neural networks make decisions
    
    ### **Domain Experts**
    - Healthcare professionals using ML models for diagnosis
    - Financial analysts relying on predictive models
    - Any domain expert who needs to understand and trust AI predictions
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## ✨ Why This App is Beneficial
    
    ### **1. Exact Interpretability**
    - Unlike approximation methods (LIME, SHAP), Re3 provides **mathematically exact** explanations
    - No sampling or approximation errors - you get the true feature contributions
    
    ### **2. Mechanistic Understanding**
    - Understand the **internal mechanism** of how ReLU networks process information
    - See how activation patterns create distinct linear regions in the input space
    - Gain insights into the geometric structure of neural network decision boundaries
    
    ### **3. Actionable Insights**
    - Identify which features are most important for specific predictions
    - Understand why a model made a particular decision
    - Debug model behavior and identify potential biases
    
    ### **4. Regulatory Compliance**
    - Provide transparent explanations for regulatory requirements (GDPR, AI Act)
    - Build trust with stakeholders through clear, interpretable results
    - Document model decision-making processes
    
    ### **5. Educational Value**
    - Visualize complex neural network concepts in an intuitive way
    - Learn how ReLU activations create piecewise-linear functions
    - Understand the relationship between network architecture and interpretability
    
    ### **6. Research Tool**
    - Analyze region properties across different model architectures
    - Compare interpretability across different datasets
    - Study the relationship between region structure and model performance
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 🔬 Technical Foundation
    
    Re3 is based on the mathematical property that ReLU networks implement **piecewise-affine functions**. 
    Each distinct pattern of ReLU activations defines a linear region where the network behaves as a 
    simple affine transformation. By identifying these regions and computing their corresponding affine 
    maps, we can provide exact explanations for any prediction.
    
    ### Key Concepts:
    - **Activation Pattern**: The binary pattern of which ReLUs are active/inactive
    - **Linear Region**: A region in input space where all samples share the same activation pattern
    - **Affine Map**: The linear transformation (A_r, D_r) that maps inputs to outputs within a region
    - **Feature Contributions**: The exact contribution of each input feature to the final prediction
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 📚 Additional Resources
    
    - **Documentation**: See `QUICK_START_GUIDE.md` for detailed usage instructions
    - **Jupyter Notebooks**: Explore example notebooks for deeper understanding
    - **Research Papers**: Refer to the underlying research on ReLU region analysis
    

    """)
    
    st.markdown("---")

    st.markdown("""
    ## 📧 Application Development Contact & Support
    
    For questions, suggestions, or collaboration opportunities:
    - **Name**: Ricky Stanley D Cruze
    - **Email**: rickystanley.dcruze@afry.com
    - **Institution**: AFRY Digital Solutions AB, Västerås
    - **Purpose**: Research and Educational Use

    ## 📧 Research Paper Contact & Support
    
    For questions, suggestions, or collaboration opportunities:
    - **Email**: arnab.barua@mdu.se
    - **Institution**: Mälardalen University
    - **Paper**: https://link.springer.com/article/10.1007/s10994-025-06957-0
    """)
    

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Re3: ReLU Region Reason - Mechanistic Interpretability Tool</p>
    <p>Developed for research purposes | AFRY Digital Solutions AB, Västerås | Mälardalen University</p>
</div>
""", unsafe_allow_html=True)

