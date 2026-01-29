# Re3 Interactive Application

An interactive web application for analyzing and interpreting ReLU neural networks using the Re3 (ReLU Region Reason) method. This tool provides exact, mathematically-grounded explanations for neural network predictions through piecewise-affine mapping.

**Note**: This application is developed by Ricky Stanley D Cruze (AFRY Digital Solutions AB) and is based on the research paper by Arnab Barua et al. (Mälardalen University).

## Features

- 🧠 **Model Loading**: Upload your trained PyTorch models or create new ones
- 📊 **Dataset Support**: Load custom datasets or use built-in datasets (Iris, Seeds, Spambase)
- 🔍 **Single Sample Analysis**: 
  - View feature-level contributions (exact, not approximate)
  - Analyze neuron contributions across all layers
  - Explore region-specific explanations
  - Visualize class probabilities with detailed breakdowns
- 📈 **Region Analysis**:
  - Identify activation regions across the dataset
  - Analyze region purity and accuracy metrics
  - Visualize region size distributions
  - Quality analysis with interactive plots
- 📋 **Model Information**: View model architecture and parameters
- 🤖 **AI-Powered Explanations**: GPT-4.1 mini explanations via OpenRouter for deeper insights
- ℹ️ **About Tab**: Comprehensive project documentation and use cases

## Installation

### Step 1: Install UV (Recommended)

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

### Step 2: Install Dependencies

After installing UV, install the project dependencies:

```bash
uv pip install -r requirments.txt
```

**Note:** If you prefer using traditional pip, you can still use:
```bash
pip install -r requirments.txt
```

### Step 3: Configure OpenRouter API (Optional)

For AI-powered explanations, create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenRouter API key
# Get your API key from https://openrouter.ai/keys
OPENROUTER_API_KEY=your_api_key_here
```

**Note**: The AI explanation feature is optional. The app works without it, but explanations will show a configuration message.

### Step 4: Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Quick Start

### Training Models

Use the interactive training script to create models for different datasets:

```bash
python quick_start.py
```

This script supports:
- **Iris Dataset** → `trained_iris_model_full.pth`
- **Accelerometer/Gyro Dataset** → `trained_accelerometer_model_full.pth`
- **Diabetes Dataset** → `trained_diabetes_model_full.pth`
- **Train All** → Train models for all datasets at once

The script will:
- Automatically configure appropriate model architectures for each dataset
- Train models with optimized hyperparameters
- Save both state dict and full model files
- Display test accuracy for each trained model

## Usage Guide

### Step 1: Load a Model

You have three options:

1. **Upload Model File**: Upload a saved PyTorch model (.pth, .pt, or .pkl file)
   - Supports full models (saved with `torch.save(model, ...)`)
   - Supports state dicts (saved with `torch.save(model.state_dict(), ...)`)
   - Automatically handles PyTorch 2.6+ security requirements
   
2. **Create New Model**: Configure and create a new model with custom architecture
   - Specify input size, hidden layer sizes, and output size
   - Supports 1-3 hidden layers
   - Can load state dicts if previously uploaded

3. **Use Pre-trained**: Use models trained with `quick_start.py`

**Note**: The app automatically extracts model parameters and validates compatibility.

### Step 2: Load a Dataset

You have two options:

1. **Upload Dataset**: Upload your own CSV or TXT file
   - For CSV files: Select the target column from a dropdown
   - For TXT files: Automatically handles tab or space-separated format
   - **Automatic Feature Cleaning**: Non-numeric columns (like timestamps, IDs) are automatically excluded
   - The app will inform you which columns were dropped

2. **Use Built-in**: Choose from:
   - **Iris**: 4 features, 3 classes (classification)
   - **Seeds**: 7 features, 3 classes (classification)
   - **Spambase**: 57 features, 2 classes (binary classification)

### Step 3: Analyze

#### Overview Tab
- Project overview and introduction
- Current status of loaded model and dataset
- Quick reference guide

#### Single Sample Analysis Tab
- Select a sample from the test set using the slider or random sample button
- View:
  - True vs predicted labels with confidence scores
  - Feature contributions (logit and probability levels) with sorting
  - Neuron contributions for each hidden layer
  - Region ID and activation pattern
  - Class probabilities with visualizations
  - Interactive bar charts and plots
  - **AI Explanation**: Get GPT-4.1 mini explanation of the prediction

#### Region Analysis Tab
- Click "Analyze All Regions" to process the entire test set
- View:
  - Summary statistics (unique regions, average purity, accuracy, total samples)
  - Detailed region statistics table
  - Region size distribution histogram
  - Purity vs accuracy scatter plot with region size indicators
  - **AI Explanation**: Get GPT-4.1 mini analysis of region patterns

#### Model Info Tab
- View complete model architecture
- Inspect weights and biases for each layer
- See feature and class information
- Understand model structure and dimensions
- **AI Explanation**: Get GPT-4.1 mini explanation of model architecture and interpretability implications

#### About Tab
- Comprehensive project description
- Detailed explanation of what the app does
- Target audience information
- Benefits and use cases
- Technical foundation and key concepts
- Contact information and resources

## Supported Datasets

### Built-in Datasets

1. **Iris** (`iris.csv`)
   - 4 features, 3 classes
   - Classic classification benchmark
   - Small dataset for quick testing

2. **Seeds** (`seeds_dataset.txt`)
   - 7 features, 3 classes
   - Agricultural classification task
   - Handles inconsistent field formatting

3. **Spambase** (`spambase.data`)
   - 57 features, 2 classes
   - Email spam detection
   - Larger feature space example

### Custom Dataset Support

The app supports uploading custom datasets with:
- **CSV files**: Any CSV with numeric features and a target column
- **TXT files**: Tab or space-separated data files
- **Automatic preprocessing**:
  - Non-numeric columns are automatically excluded
  - Missing values are handled
  - Automatic scaling and normalization
  - Label encoding for categorical targets

### Example Custom Datasets

1. **Accelerometer/Gyro Mobile Phone Dataset** (`accelerometer_gyro_mobile_phone_dataset.csv`)
   - 6 features (accX, accY, accZ, gyroX, gyroY, gyroZ)
   - Activity classification
   - Timestamp column automatically excluded

2. **Diabetes Binary Health Indicators** (`diabetes_binary_health_indicators_BRFSS2015.csv`)
   - 21 features, 2 classes (binary classification)
   - Large dataset (253K+ samples)
   - Healthcare application example

## Supported Model Architectures

The application supports ReLU neural networks with:
- **1 to 3 hidden layers**
- **Any number of neurons per layer**
- **Any number of input features**
- **Any number of output classes**
- **Sequential architecture** with Linear and ReLU layers

### Model Requirements

- Must be a PyTorch `nn.Sequential` model
- Layers must alternate: Linear → ReLU → Linear → ReLU → ... → Linear (output)
- ReLU activations only (no other activation functions)
- Output layer should not have activation (raw logits)

## File Structure

```
.
├── app.py                          # Main Streamlit application
├── re3_core.py                     # Core Re3 computation functions
├── quick_start.py                  # Interactive model training script
├── requirments.txt                 # Python dependencies
├── .gitignore                      # Git ignore rules
├── README.md                       # Project overview
├── APP_README.md                   # This file (application documentation)
├── QUICK_START_GUIDE.md            # Quick start guide
├── CONTRIBUTING.md                 # Contribution guidelines
├── iris.csv                        # Iris dataset
├── seeds_dataset.txt               # Seeds dataset
├── spambase.data                   # Spambase dataset
├── accelerometer_gyro_mobile_phone_dataset.csv  # Accelerometer dataset
└── diabetes_binary_health_indicators_BRFSS2015.csv  # Diabetes dataset
```

## Example Workflows

### Workflow 1: Iris Dataset (Quick Start)

1. **Train a model**:
   ```bash
   python quick_start.py
   # Select option 1 for Iris
   ```

2. **Start the app**: `streamlit run app.py`

3. **Load model**: 
   - In sidebar: "Upload Model File" → Upload `trained_iris_model_full.pth`

4. **Load dataset**:
   - In sidebar: "Use Built-in" → Select "Iris" → Click "Load Built-in Dataset"

5. **Analyze**:
   - Go to "Single Sample Analysis" tab
   - Select different samples and explore explanations
   - Go to "Region Analysis" tab to see overall patterns

### Workflow 2: Accelerometer Dataset

1. **Train a model**:
   ```bash
   python quick_start.py
   # Select option 2 for Accelerometer
   ```

2. **Start the app**: `streamlit run app.py`

3. **Load model**: Upload `trained_accelerometer_model_full.pth`

4. **Load dataset**:
   - "Upload Dataset" → Upload `accelerometer_gyro_mobile_phone_dataset.csv`
   - Select "Activity" as target column
   - Note: Timestamp column will be automatically excluded

5. **Analyze**: Explore feature contributions and region patterns

### Workflow 3: Diabetes Dataset

1. **Train a model**:
   ```bash
   python quick_start.py
   # Select option 3 for Diabetes
   ```

2. **Start the app**: `streamlit run app.py`

3. **Load model**: Upload `trained_diabetes_model_full.pth`

4. **Load dataset**:
   - "Upload Dataset" → Upload `diabetes_binary_health_indicators_BRFSS2015.csv`
   - Select "Diabetes_binary" as target column

5. **Analyze**: 
   - Note: Region analysis may take longer due to large dataset size
   - Explore binary classification explanations

## Tips

- **For best results**: Train your model first using `quick_start.py` or Jupyter notebooks, then load the trained model
- **Large datasets**: The region analysis may take time for large datasets (e.g., Diabetes with 253K+ samples). Consider using a subset for faster analysis
- **Model compatibility**: The app works best with Sequential PyTorch models with Linear and ReLU layers
- **Feature selection**: Non-numeric columns are automatically excluded - check the info message to see which columns were dropped
- **Class indices**: The app automatically handles sparse class indices and ensures proper indexing

## Troubleshooting

### Model not loading
- Ensure your model file is a valid PyTorch model (.pth, .pt, or .pkl)
- Check that the model architecture matches the expected format (Sequential with Linear and ReLU layers)
- For PyTorch 2.6+, the app automatically handles `weights_only=False` for full model loading
- If loading state dict, create the model architecture first, then the state dict will be applied

### Dataset loading errors
- For CSV files: Ensure the target column is clearly identified and contains valid labels
- For TXT files: The app tries whitespace-separated first, then tab-separated
- Non-numeric columns are automatically excluded - this is expected behavior
- Ensure all feature columns are numeric (or will be automatically dropped)
- Check that the target column has valid class labels

### IndexError or TypeError
- These errors have been fixed in recent updates
- Ensure you're using the latest version of the app
- If issues persist, check that class indices match the number of output classes

### Visualization issues
- If plots don't display, try refreshing the page
- Ensure matplotlib backend is properly configured
- Check browser console for JavaScript errors

### Performance issues
- Large datasets (100K+ samples) may take time for region analysis
- Consider using a subset of data for faster exploration
- Single sample analysis is always fast regardless of dataset size

### AI Explanation issues
- **"OpenRouter API key not configured"**: Create a `.env` file with `OPENROUTER_API_KEY=your_key`
- **API errors**: Check your OpenRouter account balance and API key validity
- **Slow responses**: AI explanations may take a few seconds - be patient
- **Feature works without API**: The app functions normally without the API key, just without AI explanations

## Recent Updates

### Version 1.0 Updates
- ✅ Added UV installation support for faster dependency management
- ✅ Added support for Accelerometer/Gyro and Diabetes datasets
- ✅ Enhanced `quick_start.py` with interactive dataset selection
- ✅ Added About tab with comprehensive project documentation
- ✅ Automatic non-numeric column exclusion (handles timestamps, IDs, etc.)
- ✅ Improved error handling for IndexError and TypeError
- ✅ Better bounds checking for class indices
- ✅ Support for PyTorch 2.6+ model loading
- ✅ Enhanced dataset preprocessing with automatic feature cleaning
- ✅ Improved class name handling for sparse class indices

## Citation

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

## Contact

### Application Development
For questions about the application, bugs, or feature requests:
- **Developer**: Ricky Stanley D Cruze
- **Email**: rickystanley.dcruze@afry.com
- **Institution**: AFRY Digital Solutions AB, Västerås, Sweden

### Research & Paper
For questions about the research methodology or paper:
- **Lead Researcher**: Arnab Barua
- **Email**: arnab.barua@mdu.se
- **Institution**: Mälardalen University, Sweden
- **Paper**: [Mechanistic Interpretability of ReLU Neural Networks Through Piecewise-Affine Mapping](https://link.springer.com/article/10.1007/s10994-025-06957-0)

## License

This code is provided for academic and research purposes.

---

**Re3: ReLU Region Reason** - Making Neural Networks Transparent, One Region at a Time

*Application developed by Ricky Stanley D Cruze (AFRY Digital Solutions AB) | Based on research by Arnab Barua et al. (Mälardalen University)*
