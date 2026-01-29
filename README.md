# Re3: ReLU Region Reason - Interactive Interpretability Tool

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

An interactive web application for analyzing and interpreting ReLU neural networks using the Re3 (ReLU Region Reason) method. This tool provides **exact, mathematically-grounded explanations** for neural network predictions through piecewise-affine mapping.

## 📖 About

**Re3 (ReLU Region Reason)** is an interactive interpretability tool for understanding ReLU-based neural networks through piecewise-affine mapping. Unlike approximation methods (LIME, SHAP), Re3 provides **exact explanations** by identifying specific linear regions and computing precise affine transformations.

### Key Features

- 🎯 **Exact Interpretability**: Mathematically exact explanations (no approximations)
- 🔍 **Single Sample Analysis**: Precise feature-level and neuron-level contributions
- 📈 **Region Analysis**: Automatic identification and analysis of linear regions
- 📊 **Interactive Visualizations**: Intuitive plots and charts
- 🤖 **AI-Powered Explanations**: GPT-4.1 mini explanations via OpenRouter for deeper insights

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/rickystanley76/Re3-ReLU-Region-Reason.git
cd Re3-ReLU-Region-Reason

# Install UV (recommended)
# Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -r requirments.txt
# or: pip install -r requirments.txt
```

### Usage

```bash
# Train a model (optional but recommended)
python quick_start.py
```

The `quick_start.py` script provides an interactive way to train ReLU neural network models on three pre-configured datasets:
- **Iris**: 4 features, 3 classes, 1 hidden layer
- **Accelerometer/Gyro**: 6 features, multiple classes, 1 hidden layer  
- **Diabetes**: 21 features, 2 classes (binary), 2 hidden layers

The script automatically:
- Loads and preprocesses datasets (handles various CSV formats)
- Creates appropriately-sized neural network architectures
- Trains models with optimal hyperparameters
- Saves models in two formats (state dict and full model)
- Displays training progress and test accuracy

```bash
# Launch the application
streamlit run app.py
```

The app will open at `http://localhost:8501`

**Optional**: For AI-powered explanations, create a `.env` file with your OpenRouter API key:
```bash
OPENROUTER_API_KEY=your_api_key_here
```
Get your API key from [OpenRouter](https://openrouter.ai/keys)

## 📄 Research Paper

This application is **based on** the research paper:

**"Mechanistic Interpretability of ReLU Neural Networks Through Piecewise-Affine Mapping"**

- **Authors**: Arnab Barua, Mobyen Uddin Ahmed, Shahina Begum
- **Journal**: Machine Learning (Springer)
- **DOI**: [10.1007/s10994-025-06957-0](https://link.springer.com/article/10.1007/s10994-025-06957-0)

### Citation

```bibtex
@article{barua2025re3,
  title={Mechanistic Interpretability of ReLU Neural Networks Through Piecewise-Affine Mapping},
  author={Barua, Arnab and Ahmed, Mobyen Uddin and Begum, Shahina},
  journal={Machine Learning},
  year={2025},
  doi={10.1007/s10994-025-06957-0}
}
```

## 👥 Credits

### Application Developer

**Ricky Stanley D Cruze**
- **Email**: rickystanley.dcruze@afry.com
- **Institution**: AFRY Digital Solutions AB, Västerås, Sweden
- **Role**: Application Developer

### Research Team

**Arnab Barua** (Research Lead & Paper Author)
- **Email**: arnab.barua@mdu.se
- **Institution**: Mälardalen University, Sweden

**Mobyen Uddin Ahmed** & **Shahina Begum** (Co-authors)
- **Institution**: Mälardalen University, Sweden

## 📚 Documentation

- **[QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)**: Detailed installation and usage guide
- **[APP_README.md](APP_README.md)**: Comprehensive application documentation
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Contribution guidelines

## 📁 Project Structure

Only these files (and your CSV datasets) are intended to be in the repository:

```
Re3-ReLU-Region-Reason/
├── app.py                          # Main Streamlit application
├── re3_core.py                     # Core Re3 computation functions
├── quick_start.py                  # Model training script
├── requirments.txt                 # Python dependencies
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
├── APP_README.md                   # Application documentation
├── QUICK_START_GUIDE.md            # Quick start guide
├── CONTRIBUTING.md                 # Contribution guidelines
└── [datasets]                      # CSV dataset files (e.g. iris_training_set.csv, *.csv)
```

**Uploading to GitHub:** To push only the Project Structure files and CSVs, see **"Uploading Only Project Structure Files and CSVs"** in [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) (under Publishing to GitHub).

## 📧 Contact

- **Application Issues**: rickystanley.dcruze@afry.com
- **Research Questions**: arnab.barua@mdu.se

## 📝 License

This code is provided for academic and research purposes.

---

**Re3: ReLU Region Reason** - Making Neural Networks Transparent, One Region at a Time

*Application developed by Ricky Stanley D Cruze (AFRY) | Based on research by Arnab Barua et al. (Mälardalen University)*
