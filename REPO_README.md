# Re3: ReLU Region Reason - Interactive Interpretability Tool

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Research-green.svg)](LICENSE)

An interactive web application for analyzing and interpreting ReLU neural networks using the Re3 (ReLU Region Reason) method. This tool provides **exact, mathematically-grounded explanations** for neural network predictions through piecewise-affine mapping.

## 📖 About the Application

**Re3 (ReLU Region Reason)** is an interactive interpretability tool designed for understanding and explaining the behavior of ReLU-based neural networks through piecewise-affine mapping. Rectified linear unit (ReLU) based neural networks (NNs) are recognized for their remarkable accuracy. However, the decision-making processes of these networks are often complex and difficult to understand.

Our Re3 application provides exact, mathematically-grounded explanations for neural network predictions by decomposing the network's computation into distinct linear regions defined by ReLU activation patterns.

Unlike black-box interpretability methods that provide approximate explanations (e.g., LIME, SHAP), Re3 offers **exact explanations** by identifying the specific linear region in which each input sample resides and computing the precise affine transformation that maps inputs to outputs within that region.

### Key Features

- 🎯 **Exact Interpretability**: Mathematically exact explanations (no approximations)
- 🔍 **Single Sample Analysis**: Precise feature-level and neuron-level contributions
- 📈 **Region Analysis**: Automatic identification and analysis of linear regions
- 🧠 **Model Understanding**: Deep insights into network architecture and behavior
- 📊 **Interactive Visualizations**: Intuitive plots and charts for exploration
- 🔬 **Mechanistic Understanding**: Understand the internal mechanism of ReLU networks

### What This App Does

#### 1. Single Sample Analysis
- **Exact Explanations**: Get precise feature-level contributions for individual predictions
- **Neuron-Level Insights**: Understand which neurons in each layer contribute most to the prediction
- **Region Identification**: Identify the specific activation pattern (region) for each sample
- **Probability Decomposition**: See how each feature affects both logit and probability outputs

#### 2. Region Analysis
- **Region Discovery**: Automatically identify all unique linear regions in the input space
- **Region Statistics**: Analyze region purity, accuracy, and size distribution
- **Quality Metrics**: Evaluate how well regions correspond to class boundaries
- **Visualization**: Explore region characteristics through interactive plots

#### 3. Model Understanding
- **Architecture Inspection**: View model weights, biases, and layer configurations
- **Feature Analysis**: Understand how features are used across different regions
- **Class Behavior**: Analyze how different classes are represented in the region space

### Technical Foundation

Re3 is based on the mathematical property that ReLU networks implement **piecewise-affine functions**. Each distinct pattern of ReLU activations defines a linear region where the network behaves as a simple affine transformation. By identifying these regions and computing their corresponding affine maps, we can provide exact explanations for any prediction.

**Key Concepts:**
- **Activation Pattern**: The binary pattern of which ReLUs are active/inactive
- **Linear Region**: A region in input space where all samples share the same activation pattern
- **Affine Map**: The linear transformation (A_r, D_r) that maps inputs to outputs within a region
- **Feature Contributions**: The exact contribution of each input feature to the final prediction

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/rickystanley76/Re3-ReLU-Region-Reason.git
cd Re3-ReLU-Region-Reason
```

### Step 2: Install UV (Recommended)

**UV** is a fast Python package installer. First, install UV:

**On Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**On macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Alternative: Using pip (if UV installation fails):**
```bash
pip install uv
```

### Step 3: Install Dependencies

After installing UV, install the project dependencies:

```bash
uv pip install -r requirments.txt
```

**Note:** If you prefer using traditional pip, you can still use:
```bash
pip install -r requirments.txt
```

### Step 4: Verify Installation

```bash
python -c "import streamlit, torch, pandas, numpy; print('All dependencies installed successfully!')"
```

## 💻 Usage on Local Machine

### Quick Start

1. **Train a Model (Optional but Recommended)**

   Use the interactive training script to create models:

   ```bash
   python quick_start.py
   ```

   This script supports:
   - **Iris Dataset** → `trained_iris_model_full.pth`
   - **Accelerometer/Gyro Dataset** → `trained_accelerometer_model_full.pth`
   - **Diabetes Dataset** → `trained_diabetes_model_full.pth`
   - **Train All** → Train models for all datasets at once

2. **Launch the Application**

   ```bash
   streamlit run app.py
   ```

   The app will automatically open in your default web browser at `http://localhost:8501`

### Using the Application

#### Step 1: Load a Model

You have three options:
- **Upload Model File**: Upload a trained PyTorch model (.pth, .pt, or .pkl file)
- **Create New Model**: Configure and create a new model with custom architecture
- **Use Pre-trained**: Use models trained with `quick_start.py`

#### Step 2: Load a Dataset

You have two options:
- **Upload Dataset**: Upload your own CSV or TXT file
- **Use Built-in**: Choose from Iris, Seeds, or Spambase datasets

#### Step 3: Analyze

- **Single Sample Analysis**: Select samples and view exact feature contributions
- **Region Analysis**: Analyze all regions across the dataset
- **Model Info**: Inspect model architecture and parameters

### Example Workflow

```bash
# 1. Install dependencies
uv pip install -r requirments.txt

# 2. Train a model (optional)
python quick_start.py
# Select option 1 for Iris dataset

# 3. Launch the app
streamlit run app.py
```

Then in the app:
1. Upload `trained_iris_model_full.pth` (or use built-in Iris dataset)
2. Load the Iris dataset (built-in option)
3. Explore single sample analysis and region analysis

For detailed usage instructions, see [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) and [APP_README.md](APP_README.md).

## 📚 Supported Datasets

- **Iris**: 4 features, 3 classes (classification)
- **Seeds**: 7 features, 3 classes (classification)
- **Spambase**: 57 features, 2 classes (binary classification)
- **Accelerometer/Gyro**: 6 features, activity classification
- **Diabetes**: 21 features, 2 classes (binary classification)
- **Custom Datasets**: Upload your own CSV or TXT files

## 🏗️ Supported Model Architectures

The application supports ReLU neural networks with:
- **1 to 3 hidden layers**
- **Any number of neurons per layer**
- **Any number of input features**
- **Any number of output classes**
- **Sequential architecture** with Linear and ReLU layers

## 📄 Research Paper

This application is based on the following research paper:

**Title**: Mechanistic Interpretability of ReLU Neural Networks Through Piecewise-Affine Mapping

**Authors**: 
- Arnab Barua
- Mobyen Uddin Ahmed
- Shahina Begum

**Journal**: Machine Learning (Springer)

**DOI**: [10.1007/s10994-025-06957-0](https://link.springer.com/article/10.1007/s10994-025-06957-0)

**Paper Link**: https://link.springer.com/article/10.1007/s10994-025-06957-0

### Citation

If you use this application in your research, please cite:

```bibtex
@article{barua2025re3,
  title={Mechanistic Interpretability of ReLU Neural Networks Through Piecewise-Affine Mapping},
  author={Barua, Arnab and Ahmed, Mobyen Uddin and Begum, Shahina},
  journal={Machine Learning},
  year={2025},
  doi={10.1007/s10994-025-06957-0}
}
```

## 👥 Application Development

### Developer

**Ricky Stanley D Cruze**

- **Email**: rickystanley.dcruze@afry.com
- **Institution**: AFRY Digital Solutions AB, Västerås, Sweden
- **Role**: Application Developer

### Research Team

**Arnab Barua** (Research Lead)
- **Email**: arnab.barua@mdu.se
- **Institution**: Mälardalen University, Sweden

**Mobyen Uddin Ahmed** (Co-author)
- **Institution**: Mälardalen University, Sweden

**Shahina Begum** (Co-author)
- **Institution**: Mälardalen University, Sweden

## 📁 Project Structure

```
Re3-ReLU-Region-Reason/
├── app.py                          # Main Streamlit application
├── re3_core.py                     # Core Re3 computation functions
├── quick_start.py                  # Interactive model training script
├── requirments.txt                 # Python dependencies
├── .gitignore                      # Git ignore rules
├── README.md                       # Project overview (notebooks)
├── REPO_README.md                  # This file (GitHub repository README)
├── APP_README.md                   # Application documentation
├── QUICK_START_GUIDE.md            # Quick start guide
├── CONTRIBUTING.md                 # Contribution guidelines
├── iris.csv                        # Iris dataset
├── seeds_dataset.txt               # Seeds dataset
├── spambase.data                   # Spambase dataset
├── accelerometer_gyro_mobile_phone_dataset.csv  # Accelerometer dataset
└── diabetes_binary_health_indicators_BRFSS2015.csv  # Diabetes dataset
```

## 🔧 Requirements

- Python >= 3.8
- pandas >= 1.3.0
- numpy >= 1.21.0
- torch >= 1.9.0
- scikit-learn >= 0.24.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- streamlit >= 1.28.0
- seaborn >= 0.12.0

See `requirments.txt` for the complete list of dependencies.

## 📖 Documentation

- **[QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)**: Quick start guide with installation and usage instructions
- **[APP_README.md](APP_README.md)**: Comprehensive application documentation
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Guidelines for contributing to the project

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📝 License

This code is provided for academic and research purposes.

## 🙏 Acknowledgments

- **Research**: This work is based on research conducted at the School of Innovation, Design and Engineering, Mälardalen University, Sweden
- **Development**: Application developed at AFRY Digital Solutions AB, Västerås, Sweden
- **Funding**: Research and development supported by AFRY Digital Solutions AB and Mälardalen University

## 📧 Contact & Support

### Application Development
- **Developer**: Ricky Stanley D Cruze
- **Email**: rickystanley.dcruze@afry.com
- **Institution**: AFRY Digital Solutions AB, Västerås, Sweden

### Research & Paper
- **Lead Researcher**: Arnab Barua
- **Email**: arnab.barua@mdu.se
- **Institution**: Mälardalen University, Sweden
- **Paper**: [Mechanistic Interpretability of ReLU Neural Networks Through Piecewise-Affine Mapping](https://link.springer.com/article/10.1007/s10994-025-06957-0)

## 🌟 Features Highlights

- ✅ **Exact Interpretability**: No approximations - mathematically exact explanations
- ✅ **Interactive Web Interface**: User-friendly Streamlit application
- ✅ **Multiple Dataset Support**: Works with various classification datasets
- ✅ **Comprehensive Analysis**: Single sample and region-level analysis
- ✅ **Visualization**: Rich interactive plots and charts
- ✅ **Well Documented**: Extensive documentation and examples

## 🚀 Quick Links

- [Quick Start Guide](QUICK_START_GUIDE.md)
- [Application Documentation](APP_README.md)
- [Research Paper](https://link.springer.com/article/10.1007/s10994-025-06957-0)
- [GitHub Repository](https://github.com/rickystanley76/Re3-ReLU-Region-Reason)

---

**Re3: ReLU Region Reason** - Making Neural Networks Transparent, One Region at a Time

*Developed for research purposes | AFRY Digital Solutions AB, Västerås | Mälardalen University*
