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
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from re3_core import (
    extract_model_parameters,
    compute_region_affine_1layer,
    compute_region_affine_2layer,
    compute_region_affine_3layer,
    explain_region,
    analyze_regions
)

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
    """Load dataset from file."""
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path)
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
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
    model_option = st.radio(
        "Model Source",
        ["Upload Model File", "Create New Model", "Use Pre-trained"],
        key="model_option"
    )
    
    if model_option == "Upload Model File":
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
                st.info("💡 State dict loaded. Use 'Create New Model' with matching architecture, then the state dict will be applied automatically.")
                st.session_state.model_state_dict = state_dict
    
    elif model_option == "Create New Model":
        st.info("Configure model architecture:")
        input_size = st.number_input("Input Size", min_value=1, value=4, step=1)
        num_layers = st.number_input("Number of Hidden Layers", min_value=1, max_value=3, value=1, step=1)
        
        hidden_sizes = []
        for i in range(num_layers):
            size = st.number_input(f"Hidden Layer {i+1} Size", min_value=1, value=8, step=1, key=f"hidden_{i}")
            hidden_sizes.append(size)
        
        output_size = st.number_input("Output Size (Classes)", min_value=2, value=3, step=1)
        
        if st.button("Create Model"):
            model = create_model_from_config(input_size, hidden_sizes, output_size, num_layers)
            
            # If we have a state dict, try to load it
            if 'model_state_dict' in st.session_state:
                try:
                    model.load_state_dict(st.session_state.model_state_dict)
                    st.success("✅ Model created and loaded with saved weights!")
                except Exception as e:
                    st.warning(f"⚠️ Could not load state dict: {str(e)}. Using random initialization.")
                    st.success("✅ Model created with random weights!")
            else:
                st.success("✅ Model created with random weights!")
            
            st.session_state.model = model
            st.session_state.model_params = extract_model_parameters(model)
    
    # Dataset loading section
    st.subheader("2. Load Dataset")
    dataset_option = st.radio(
        "Dataset Source",
        ["Upload Dataset", "Use Built-in"],
        key="dataset_option"
    )
    
    if dataset_option == "Upload Dataset":
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
    
    elif dataset_option == "Use Built-in":
        builtin_datasets = {
            "Iris": "iris.csv",
            "Seeds": "seeds_dataset.txt",
            "Spambase": "spambase.data"
        }
        
        selected_dataset = st.selectbox("Select Dataset", list(builtin_datasets.keys()))
        
        if st.button("Load Built-in Dataset"):
            dataset_path = builtin_datasets[selected_dataset]
            if Path(dataset_path).exists():
                if selected_dataset == "Iris":
                    df = pd.read_csv(dataset_path)
                    data_prep = prepare_data(df, 'variety')
                elif selected_dataset == "Seeds":
                    # Handle inconsistent field counts by using whitespace separator
                    df = pd.read_csv(dataset_path, sep='\s+', header=None, engine='python')
                    # Last column is class (1-3)
                    df.columns = [f'Feature_{i+1}' for i in range(7)] + ['Class']
                    data_prep = prepare_data(df, 'Class')
                else:  # Spambase
                    df = pd.read_csv(dataset_path, header=None)
                    df.columns = [f'Feature_{i+1}' for i in range(57)] + ['Class']
                    data_prep = prepare_data(df, 'Class')
                
                st.session_state.X_test = data_prep['X_test']
                st.session_state.y_test = data_prep['y_test']
                st.session_state.feature_names = data_prep['feature_names']
                st.session_state.class_names = data_prep['class_names']
                st.session_state.scaler = data_prep['scaler']
                st.success("✅ Dataset loaded!")


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
            
            # Region size distribution
            st.subheader("Region Size Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(results['region_stats']['n_samples'], bins=20, edgecolor='black')
            ax.set_xlabel('Number of Samples per Region')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Region Sizes')
            st.pyplot(fig)
            
            # Purity vs Accuracy scatter
            st.subheader("Region Purity vs Accuracy")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(results['region_stats']['purity'], 
                      results['region_stats']['accuracy'],
                      s=results['region_stats']['n_samples']*10,
                      alpha=0.6)
            ax.set_xlabel('Purity')
            ax.set_ylabel('Accuracy')
            ax.set_title('Region Quality Analysis')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)


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

