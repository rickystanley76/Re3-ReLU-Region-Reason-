"""
Quick Start Script - Train a simple model and save it for use in the Re3 app
This script trains models on different datasets (Iris, Accelerometer/Gyro, Diabetes)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import sys
import os
from typing import Tuple

def _read_csv_auto(path: str) -> pd.DataFrame:
    """
    Read a CSV-like file with automatic delimiter detection.

    Many of our provided datasets are semicolon-delimited (';') even though they
    have a .csv extension. Using sep=None + engine='python' lets pandas infer
    the delimiter reliably (',' vs ';' etc.).
    """
    return pd.read_csv(path, sep=None, engine="python")

def load_iris_dataset():
    """Load and prepare Iris dataset"""
    print("Loading Iris dataset...")
    # Note: file is semicolon-delimited in this repo
    df = _read_csv_auto('iris_training_set.csv')
    
    # Prepare data
    # The last column should be 'variety' (Setosa, Versicolor, Virginica)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df.iloc[:, -1])
    # Drop the original variety column (keep only label)
    df = df.drop(columns=[df.columns[-2]])
    
    X = df.iloc[:, :-1].values
    y = df['label'].values
    
    return X, y, 'Iris', 4, 3

def load_accelerometer_dataset():
    """Load and prepare Accelerometer/Gyro dataset"""
    print("Loading Accelerometer/Gyro Mobile Phone dataset...")
    df = _read_csv_auto('accelerometer_Training_set.csv')
    
    # Drop timestamp column if it exists (non-numeric like 36:03.1)
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    
    # Activity is the target (last column)
    X_df = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].values

    # Ensure features are numeric (defensive)
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    # Drop any columns that became entirely NaN after coercion
    X_df = X_df.dropna(axis=1, how="all")
    # Drop rows with any NaNs
    X_df = X_df.dropna(axis=0, how="any")
    # Align y with filtered rows
    y = y[X_df.index.values]
    X = X_df.values
    
    # Encode labels if needed (in case they're not numeric)
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Get number of unique classes
    n_classes = len(np.unique(y))
    
    return X, y, 'Accelerometer', X.shape[1], n_classes

def load_diabetes_dataset():
    """Load and prepare Diabetes dataset"""
    print("Loading Diabetes Binary Health Indicators dataset...")
    # Note: file is semicolon-delimited in this repo
    df = _read_csv_auto('diabetes_Training_set.csv')
    
    # Diabetes_binary is the first column (target)
    # All other columns are features
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    
    # Convert to integer if needed
    # Data may come as floats (0.0/1.0) depending on how the CSV was produced.
    y = pd.to_numeric(y, errors="raise").astype(int)
    
    # Get number of unique classes (should be 2 for binary classification)
    n_classes = len(np.unique(y))
    
    return X, y, 'Diabetes', X.shape[1], n_classes

def create_model(input_size, hidden_size, output_size, num_hidden_layers=1):
    """Create a neural network model"""
    layers = []
    
    # Input to first hidden layer
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(nn.ReLU())
    
    # Additional hidden layers
    for _ in range(num_hidden_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
    
    # Output layer
    layers.append(nn.Linear(hidden_size, output_size))
    
    return nn.Sequential(*layers)

def train_model(model, train_loader, test_loader, epochs=100, lr=0.01):
    """Train the model"""
    print("Training model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    
    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    
    return accuracy

def main():
    """Main function to train models on different datasets"""
    print("=" * 60)
    print("Re3 Quick Start - Model Training")
    print("=" * 60)
    print("\nAvailable datasets:")
    print("1. Iris (iris_training_set.csv)")
    print("2. Accelerometer/Gyro Mobile Phone (accelerometer_Training_set.csv)")
    print("3. Diabetes Binary Health Indicators (diabetes_training_set.csv)")
    print("\n" + "=" * 60)
    
    # Dataset configuration
    dataset_configs = {
        '1': {
            'loader': load_iris_dataset,
            'hidden_size': 8,
            'num_hidden_layers': 1,
            'epochs': 100,
            'batch_size': 16
        },
        '2': {
            'loader': load_accelerometer_dataset,
            'hidden_size': 16,
            'num_hidden_layers': 1,
            'epochs': 100,
            'batch_size': 32
        },
        '3': {
            'loader': load_diabetes_dataset,
            'hidden_size': 32,
            'num_hidden_layers': 2,
            'epochs': 50,
            'batch_size': 64
        }
    }
    
    # Get user choice
    choice = input("\nSelect dataset (1-3) or 'all' to train all: ").strip().lower()
    
    if choice == 'all':
        choices = ['1', '2', '3']
    elif choice in dataset_configs:
        choices = [choice]
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)
    
    # Train models for selected datasets
    for choice in choices:
        config = dataset_configs[choice]
        
        try:
            # Load dataset
            X, y, dataset_name, input_size, output_size = config['loader']()
            
            print(f"\n{'='*60}")
            print(f"Training model for {dataset_name} dataset")
            print(f"Input size: {input_size}, Output size: {output_size}")
            print(f"{'='*60}\n")
            
            # Split and scale
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)
            
            train_loader = DataLoader(
                TensorDataset(X_train_tensor, y_train_tensor), 
                batch_size=config['batch_size'], 
                shuffle=True
            )
            test_loader = DataLoader(
                TensorDataset(X_test_tensor, y_test_tensor), 
                batch_size=config['batch_size']
            )
            
            # Create model
            print("Creating model...")
            model = create_model(
                input_size, 
                config['hidden_size'], 
                output_size, 
                config['num_hidden_layers']
            )
            print(f"Model architecture: {model}")
            
            # Train model
            accuracy = train_model(
                model, 
                train_loader, 
                test_loader, 
                epochs=config['epochs']
            )
            
            # Save model
            dataset_name_lower = dataset_name.lower()
            model_path = f'trained_{dataset_name_lower}_model.pth'
            full_model_path = f'trained_{dataset_name_lower}_model_full.pth'
            
            torch.save(model.state_dict(), model_path)
            torch.save(model, full_model_path)
            
            print(f"\n✅ Model training complete for {dataset_name}!")
            print(f"Model saved to: {model_path}")
            print(f"Full model saved to: {full_model_path}")
            print(f"Test Accuracy: {accuracy:.2f}%")
            
        except FileNotFoundError as e:
            print(f"\n❌ Error: Could not find dataset file: {e}")
            print(f"Skipping {dataset_configs[choice]['loader'].__name__}...")
        except Exception as e:
            print(f"\n❌ Error training model for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ All selected models trained!")
    print("\nTo use in the Re3 app:")
    print("1. Run: streamlit run app.py")
    print("2. In the sidebar, select 'Upload Model File'")
    print("3. Upload the trained model file (e.g., 'trained_iris_model_full.pth')")
    print("4. Load the corresponding dataset")
    print("5. Start analyzing!")
    print("=" * 60)

if __name__ == "__main__":
    main()
