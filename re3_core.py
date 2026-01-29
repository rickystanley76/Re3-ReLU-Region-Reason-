"""
Re3 Core Module - ReLU Region Reason
Provides functions for computing region-specific affine maps and explanations
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import softmax
from typing import Tuple, Dict, List, Optional


def extract_model_parameters(model: nn.Module) -> Dict:
    """
    Extract weights and biases from a PyTorch Sequential model.
    
    Args:
        model: PyTorch Sequential model with Linear and ReLU layers
        
    Returns:
        Dictionary with weights and biases for each layer
        num_layers: number of hidden layers (excluding output layer)
    """
    params = {}
    layer_idx = 0
    
    for i, module in enumerate(model):
        if isinstance(module, nn.Linear):
            params[f'w{layer_idx}'] = module.weight.data.cpu().numpy()
            params[f'b{layer_idx}'] = module.bias.data.cpu().numpy()
            layer_idx += 1
    
    # num_layers represents hidden layers only (total layers - 1 for output layer)
    if layer_idx == 0:
        raise ValueError("Model has no Linear layers")
    if layer_idx == 1:
        raise ValueError("Model must have at least one hidden layer. Found only output layer.")
    
    params['num_layers'] = layer_idx - 1
    return params


def compute_region_affine_1layer(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, 
                                  w2: np.ndarray, b2: np.ndarray) -> Tuple:
    """
    Compute region-specific affine map for 1 hidden layer network.
    
    Args:
        x: Input vector (d,)
        w1: Hidden layer weights (h, d)
        b1: Hidden layer biases (h,)
        w2: Output layer weights (c, h)
        b2: Output layer biases (c,)
        
    Returns:
        A_r: Effective weight matrix (c, d)
        D_r: Effective bias vector (c,)
        logits: Output logits (c,)
        pred: Predicted class index
        c1: Hidden layer neuron contributions (h,)
        region_id: Region identifier string
    """
    # Hidden layer
    Z1 = w1.dot(x) + b1
    S1 = (Z1 > 0).astype(float)
    H1 = np.maximum(0, Z1)
    
    # Output layer
    logits = w2.dot(H1) + b2
    pred = int(np.argmax(logits))
    
    # Collapsed affine map
    W2m = w2 * S1[None, :]
    A_r = W2m.dot(w1)
    D_r = b2 + W2m.dot(b1)
    
    # Neuron contributions
    c1 = H1 * W2m[pred]
    
    # Region ID
    S1_bits = ''.join(str(int(b)) for b in S1)
    region_id = f"L1:{S1_bits}"
    
    return A_r, D_r, logits, pred, c1, region_id


def compute_region_affine_2layer(x: np.ndarray, w1: np.ndarray, b1: np.ndarray,
                                  w2: np.ndarray, b2: np.ndarray,
                                  w3: np.ndarray, b3: np.ndarray) -> Tuple:
    """
    Compute region-specific affine map for 2 hidden layer network.
    
    Args:
        x: Input vector (d,)
        w1: First hidden layer weights (h1, d)
        b1: First hidden layer biases (h1,)
        w2: Second hidden layer weights (h2, h1)
        b2: Second hidden layer biases (h2,)
        w3: Output layer weights (c, h2)
        b3: Output layer biases (c,)
        
    Returns:
        A_r: Effective weight matrix (c, d)
        D_r: Effective bias vector (c,)
        logits: Output logits (c,)
        pred: Predicted class index
        c1: First hidden layer neuron contributions (h1,)
        c2: Second hidden layer neuron contributions (h2,)
        region_id: Region identifier string
    """
    # First hidden layer
    Z1 = w1.dot(x) + b1
    S1 = (Z1 > 0).astype(float)
    H1 = np.maximum(0, Z1)
    
    # Second hidden layer
    Z2 = w2.dot(H1) + b2
    S2 = (Z2 > 0).astype(float)
    H2 = np.maximum(0, Z2)
    
    # Output layer
    logits = w3.dot(H2) + b3
    pred = int(np.argmax(logits))
    
    # Collapsed affine map
    W3m = w3 * S2[None, :]
    W2m = w2 * S1[None, :]
    A_r = W3m.dot(W2m).dot(w1)
    D_r = b3 + W3m.dot(b2) + W3m.dot(W2m).dot(b1)
    
    # Neuron contributions
    c2 = H2 * W3m[pred]
    c1 = H1 * W2m.dot(c2)
    
    # Region ID
    S1_bits = ''.join(str(int(b)) for b in S1)
    S2_bits = ''.join(str(int(b)) for b in S2)
    region_id = f"L1:{S1_bits}_L2:{S2_bits}"
    
    return A_r, D_r, logits, pred, c1, c2, region_id


def compute_region_affine_3layer(x: np.ndarray, w1: np.ndarray, b1: np.ndarray,
                                  w2: np.ndarray, b2: np.ndarray,
                                  w3: np.ndarray, b3: np.ndarray,
                                  w4: np.ndarray, b4: np.ndarray) -> Tuple:
    """
    Compute region-specific affine map for 3 hidden layer network.
    
    Args:
        x: Input vector (d,)
        w1: First hidden layer weights (h1, d)
        b1: First hidden layer biases (h1,)
        w2: Second hidden layer weights (h2, h1)
        b2: Second hidden layer biases (h2,)
        w3: Third hidden layer weights (h3, h2)
        b3: Third hidden layer biases (h3,)
        w4: Output layer weights (c, h3)
        b4: Output layer biases (c,)
        
    Returns:
        A_r: Effective weight matrix (c, d)
        D_r: Effective bias vector (c,)
        logits: Output logits (c,)
        pred: Predicted class index
        c1: First hidden layer neuron contributions (h1,)
        c2: Second hidden layer neuron contributions (h2,)
        c3: Third hidden layer neuron contributions (h3,)
        region_id: Region identifier string
    """
    # First hidden layer
    Z1 = w1.dot(x) + b1
    S1 = (Z1 > 0).astype(float)
    H1 = np.maximum(0, Z1)
    
    # Second hidden layer
    Z2 = w2.dot(H1) + b2
    S2 = (Z2 > 0).astype(float)
    H2 = np.maximum(0, Z2)
    
    # Third hidden layer
    Z3 = w3.dot(H2) + b3
    S3 = (Z3 > 0).astype(float)
    H3 = np.maximum(0, Z3)
    
    # Output layer
    logits = w4.dot(H3) + b4
    pred = int(np.argmax(logits))
    
    # Collapsed affine map
    W4m = w4 * S3[None, :]
    W3m = w3 * S2[None, :]
    W2m = w2 * S1[None, :]
    A_r = W4m.dot(W3m).dot(W2m).dot(w1)
    D_r = b4 + W4m.dot(b3) + W4m.dot(W3m).dot(b2) + W4m.dot(W3m).dot(W2m).dot(b1)
    
    # Neuron contributions
    c3 = H3 * W4m[pred]
    c2 = H2 * W3m.dot(c3)
    c1 = H1 * W2m.dot(c2)
    
    # Region ID
    S1_bits = ''.join(str(int(b)) for b in S1)
    S2_bits = ''.join(str(int(b)) for b in S2)
    S3_bits = ''.join(str(int(b)) for b in S3)
    region_id = f"L1:{S1_bits}_L2:{S2_bits}_L3:{S3_bits}"
    
    return A_r, D_r, logits, pred, c1, c2, c3, region_id


def explain_region(x: np.ndarray, A_r: np.ndarray, D_r: np.ndarray, 
                   logits: np.ndarray, pred: int, 
                   feature_names: List[str], class_names: List[str]) -> pd.DataFrame:
    """
    Compute feature-level explanations for a sample.
    
    Args:
        x: Input vector (d,)
        A_r: Region-specific weight matrix (c, d)
        D_r: Region-specific bias (c,)
        logits: Output logits (c,)
        pred: Predicted class index
        feature_names: List of feature names
        class_names: List of class names
        
    Returns:
        DataFrame with feature contributions
    """
    # Bounds check for pred
    num_classes = A_r.shape[0]
    if pred < 0 or pred >= num_classes:
        # Use 0 as fallback if pred is out of range
        pred = 0
    
    # Logit-level contributions for predicted class
    A_c = A_r[pred]
    f_logit = A_c * x
    
    # Compute probabilities
    logits_r = A_r.dot(x) + D_r
    probs = softmax(logits_r)
    
    # Softmax Jacobian for predicted class
    J = -probs[pred] * probs
    J[pred] = probs[pred] * (1 - probs[pred])
    
    # Probability-level contributions
    f_all = A_r * x
    f_prob = J.dot(f_all)
    
    # Build DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'Value': x,
        'LogitContribution': f_logit,
        'ProbContribution': f_prob,
        'PredProbability': probs[pred] if pred < len(probs) else 0.0
    })
    
    return df


def analyze_regions(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray,
                    params: Dict, feature_names: List[str], 
                    class_names: List[str], num_layers: int) -> Dict:
    """
    Analyze regions across a dataset.
    
    Args:
        X: Input features (N, d)
        y: True labels (N,)
        y_pred: Predicted labels (N,)
        params: Model parameters dictionary
        feature_names: List of feature names
        class_names: List of class names
        num_layers: Number of hidden layers
        
    Returns:
        Dictionary with region statistics and feature profiles
    """
    regions = {}
    region_samples = {}
    
    N = len(X)
    
    for i in range(N):
        x = X[i]
        
        if num_layers == 1:
            result = compute_region_affine_1layer(x, params['w0'], params['b0'],
                                                  params['w1'], params['b1'])
            region_id = result[5]
        elif num_layers == 2:
            result = compute_region_affine_2layer(x, params['w0'], params['b0'],
                                                   params['w1'], params['b1'],
                                                   params['w2'], params['b2'])
            region_id = result[6]
        elif num_layers == 3:
            result = compute_region_affine_3layer(x, params['w0'], params['b0'],
                                                  params['w1'], params['b1'],
                                                  params['w2'], params['b2'],
                                                  params['w3'], params['b3'])
            region_id = result[7]
        else:
            raise ValueError(f"Unsupported number of layers: {num_layers}")
        
        if region_id not in regions:
            regions[region_id] = {
                'samples': [],
                'labels': [],
                'predictions': []
            }
        
        regions[region_id]['samples'].append(i)
        regions[region_id]['labels'].append(y[i])
        regions[region_id]['predictions'].append(y_pred[i])
        region_samples[region_id] = x
    
    # Compute region statistics
    region_stats = []
    for region_id, data in regions.items():
        labels = np.array(data['labels'])
        predictions = np.array(data['predictions'])
        n_samples = len(data['samples'])
        
        # Class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        majority_class = unique_labels[np.argmax(counts)]
        # Convert to int to ensure proper indexing
        majority_class_int = int(majority_class) if isinstance(majority_class, (np.floating, np.integer)) else majority_class
        purity = counts.max() / n_samples
        
        # Prediction accuracy
        accuracy = (labels == predictions).mean()
        
        # Bounds check to prevent IndexError
        if 0 <= majority_class_int < len(class_names):
            majority_class_name = class_names[majority_class_int]
        else:
            majority_class_name = f"Class_{majority_class_int}"
        
        region_stats.append({
            'region_id': region_id,
            'n_samples': n_samples,
            'purity': purity,
            'majority_class': majority_class_name,
            'accuracy': accuracy
        })
    
    region_stats_df = pd.DataFrame(region_stats).sort_values('n_samples', ascending=False)
    
    return {
        'regions': regions,
        'region_stats': region_stats_df,
        'n_unique_regions': len(regions)
    }
