# Quick Start Guide - Re3 Interactive Application

## 🚀 Getting Started in 3 Steps

### Step 1: Install UV and Dependencies

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

After installing UV, install the project dependencies:

```bash
uv pip install -r requirments.txt
```

**Note:** If you prefer using traditional pip, you can still use:
```bash
pip install -r requirments.txt
```

### Step 2: Train a Model (Optional but Recommended)

Train a model using the quick start script:

```bash
python quick_start.py
```

#### What `quick_start.py` Does

The `quick_start.py` script is a comprehensive model training utility that:

1. **Loads Datasets**: Automatically loads and preprocesses three different datasets:
   - **Iris Dataset** (`iris_training_set.csv`): 4 features (sepal length, sepal width, petal length, petal width), 3 classes (Setosa, Versicolor, Virginica)
   - **Accelerometer/Gyro Dataset** (`accelerometer_Training_set.csv`): 6 features (accX, accY, accZ, gyroX, gyroY, gyroZ), multiple activity classes
   - **Diabetes Dataset** (`diabetes_Training_set.csv`): 21 features, 2 classes (binary classification)

2. **Handles Data Formats**: 
   - Automatically detects CSV delimiters (handles both comma and semicolon-separated files)
   - Drops non-numeric columns (e.g., timestamp columns)
   - Encodes categorical labels automatically
   - Handles missing values and data type conversions

3. **Creates Neural Network Models**:
   - **Iris**: 1 hidden layer with 8 neurons
   - **Accelerometer**: 1 hidden layer with 16 neurons
   - **Diabetes**: 2 hidden layers with 32 neurons each
   - All models use ReLU activations (required for Re3 interpretability)

4. **Trains Models**:
   - Uses Adam optimizer with learning rate 0.01
   - Splits data into 80% training and 20% testing
   - Scales features using StandardScaler
   - Trains for configurable epochs (100 for Iris/Accelerometer, 50 for Diabetes)
   - Displays training progress and test accuracy

5. **Saves Models**:
   - Saves two versions of each model:
     - `trained_*_model.pth`: State dictionary only (weights and biases)
     - `trained_*_model_full.pth`: Complete model with architecture (recommended for app use)

#### Interactive Usage

When you run the script, you'll see:

```
============================================================
Re3 Quick Start - Model Training
============================================================

Available datasets:
1. Iris (iris_training_set.csv)
2. Accelerometer/Gyro Mobile Phone (accelerometer_Training_set.csv)
3. Diabetes Binary Health Indicators (diabetes_training_set.csv)

============================================================

Select dataset (1-3) or 'all' to train all: 
```

**Options**:
- Enter `1`, `2`, or `3` to train a specific dataset
- Enter `all` to train all three datasets sequentially

**Output**: The script will display:
- Dataset loading progress
- Model architecture details
- Training progress (loss every 20 epochs)
- Final test accuracy
- File paths where models are saved

**Alternative**: You can also train models using the Jupyter notebooks (`re3_1_layer.ipynb`, `re3_2_layers.ipynb`, `re3_3_layers.ipynb`)

#### Technical Details of `quick_start.py`

The script is organized into several key functions:

**Data Loading Functions**:
- `_read_csv_auto()`: Automatically detects CSV delimiters (handles both `,` and `;` separators)
- `load_iris_dataset()`: Loads Iris dataset, encodes labels, returns features and targets
- `load_accelerometer_dataset()`: Loads accelerometer data, drops timestamp column, handles numeric conversion
- `load_diabetes_dataset()`: Loads diabetes dataset, handles binary classification setup

**Model Functions**:
- `create_model()`: Creates PyTorch Sequential model with ReLU activations
- `train_model()`: Trains model using Adam optimizer and CrossEntropyLoss

**Main Workflow**:
1. User selects dataset(s) to train
2. Script loads dataset using appropriate loader function
3. Data is split (80% train, 20% test) and scaled
4. Model architecture is created based on dataset configuration
5. Model is trained with progress updates
6. Two model files are saved:
   - State dict only (for loading into pre-defined architectures)
   - Full model (includes architecture - recommended for app)

**Error Handling**:
- Handles missing dataset files gracefully
- Provides clear error messages
- Continues with other datasets if one fails (when using 'all' option)

### Step 3: Launch the Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## 📦 Available Trained Models

After running `quick_start.py`, you'll have the following model files available:

| Dataset | Model Files | Description |
|---------|------------|-------------|
| **Iris** | `trained_iris_model_full.pth`<br>`trained_iris_model.pth` | 4 features, 3 classes, 1 hidden layer |
| **Accelerometer/Gyro** | `trained_accelerometer_model_full.pth`<br>`trained_accelerometer_model.pth` | 6 features, multiple activity classes, 1 hidden layer |
| **Diabetes** | `trained_diabetes_model_full.pth`<br>`trained_diabetes_model.pth` | 21 features, 2 classes (binary), 2 hidden layers |

**Note**: Use the `*_full.pth` files for easier loading in the app (they contain the full model architecture).

## 📖 Basic Usage

### Loading a Model

1. In the sidebar, under "1. Load Model"
2. **Upload Model File**: Upload one of the trained models:
   - `trained_iris_model_full.pth` (for Iris dataset)
   - `trained_accelerometer_model_full.pth` (for Accelerometer/Gyro dataset)
   - `trained_diabetes_model_full.pth` (for Diabetes dataset)
   - Or upload your own PyTorch model (.pth, .pt, or .pkl)
3. Model status will show ✅ when loaded

### Loading a Dataset

1. In the sidebar, under "2. Load Dataset"
2. **Upload Dataset**: Upload your CSV/TXT file
   - The app automatically detects delimiters (comma or semicolon)
   - Non-numeric columns (like timestamps) are automatically excluded
3. Select the target column from the dropdown
4. Click "Prepare Dataset"
5. **Note**: The app uses 100% of your uploaded dataset for analysis (no train/test split for interpretation)

### Analyzing Samples

1. Go to **"🔍 Single Sample Analysis"** tab
2. Use the slider to select a sample
3. View:
   - Prediction vs true label
   - Feature contributions (which features matter most)
   - Neuron contributions (which neurons are important)
   - Region ID and class probabilities

### Analyzing Regions

1. Go to **"📈 Region Analysis"** tab
2. Click **"Analyze All Regions"**
3. Explore:
   - Summary metrics (unique regions, average purity, average accuracy, total samples)
   - Region statistics table with detailed metrics
   - **Sankey Diagram**: Multi-layer flow visualization
     - For 1-layer models: Shows Regions → Classes flow
     - For 2-layer models: Shows Features → L1 Regions → L2 Regions → Classes flow
   - AI-powered explanations of region patterns

## 🎯 Example Workflow

### Complete Example with Iris Dataset

```bash
# 1. Install dependencies (using UV)
uv pip install -r requirments.txt

# 2. Train a model (select option 1 for Iris)
python quick_start.py

# 3. Launch app
streamlit run app.py
```

Then in the app:

1. **Load Model**: 
   - Upload `trained_iris_model_full.pth`

2. **Load Dataset**:
   - Upload `iris_training_set.csv` (or `iris_test.csv`)
   - Select "variety" as the target column
   - Click "Prepare Dataset"

3. **Analyze**:
   - Go to "Single Sample Analysis"
   - Move the slider to see different samples
   - Observe how feature contributions change

4. **Region Analysis**:
   - Go to "Region Analysis" tab
   - Click "Analyze All Regions"
   - Explore the statistics and visualizations

### Example with Accelerometer/Gyro Dataset

```bash
# 1. Install dependencies (using UV)
uv pip install -r requirments.txt

# 2. Train a model (select option 2 for Accelerometer)
python quick_start.py

# 3. Launch app
streamlit run app.py
```

Then in the app:

1. **Load Model**: 
   - Upload `trained_accelerometer_model_full.pth`

2. **Load Dataset**:
   - Upload `accelerometer_gyro_mobile_phone_dataset.csv`
   - Select "Activity" as the target column
   - Click "Prepare Dataset"

3. **Analyze**: Follow the same steps as Iris example

### Example with Diabetes Dataset

```bash
# 1. Install dependencies (using UV)
uv pip install -r requirments.txt

# 2. Train a model (select option 3 for Diabetes)
python quick_start.py

# 3. Launch app
streamlit run app.py
```

Then in the app:

1. **Load Model**: 
   - Upload `trained_diabetes_model_full.pth`

2. **Load Dataset**:
   - Upload `diabetes_binary_health_indicators_BRFSS2015.csv`
   - Select "Diabetes_binary" as the target column
   - Click "Prepare Dataset"

3. **Analyze**: Follow the same steps as Iris example

**Note**: For Diabetes dataset, region analysis may take longer due to the large dataset size (253K+ samples).

## 💡 Tips

- **For meaningful results**: Train your model first (untrained models will give random predictions)
- **Large datasets**: Region analysis may take time - be patient
- **Custom models**: Ensure your model uses ReLU activations and Sequential architecture
- **Feature names**: The app will use column names from your dataset automatically

## 🔧 Troubleshooting

### "Model not loading"
- Check file format (.pth, .pt, or .pkl)
- Ensure it's a PyTorch Sequential model with Linear and ReLU layers
- Try using `quick_start.py` to create a compatible model

### "Dataset errors"
- For CSV: Ensure target column is clearly identified
- For TXT: Check if it's tab or space-separated
- Ensure all feature columns are numeric

### "No visualizations"
- Refresh the page
- Check browser console for errors
- Ensure matplotlib is properly installed

## 📚 Next Steps

- Explore different samples to see how explanations vary
- Try different datasets to understand region behavior
- Compare models with different architectures
- Read the full documentation in `APP_README.md`

## 🆘 Need Help?

- Check `APP_README.md` for detailed documentation
- Review the Jupyter notebooks for understanding the Re3 method
- **Application Issues**: Contact rickystanley.dcruze@afry.com (Application Developer)
- **Research Questions**: Contact arnab.barua@mdu.se (Research Lead)

---

## 📤 Publishing to GitHub and Version Control

This section explains how to publish your Re3 project to GitHub and maintain it with regular updates.

### Prerequisites

1. **Install Git**: If not already installed, download from [git-scm.com](https://git-scm.com/)
2. **Create GitHub Account**: Sign up at [github.com](https://github.com/) if you don't have one
3. **Verify Git Installation**:
   ```bash
   git --version
   ```

### Step 1: Initialize Git Repository

If your project is not already a git repository:

```bash
# Navigate to your project directory
cd /path/to/Re3-working_rd

# Initialize git repository
git init

# Configure your git identity (if not already done)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Step 2: Create GitHub Repository

1. **Go to GitHub**: Log in to [github.com](https://github.com/)
2. **Create New Repository**:
   - Click the "+" icon in the top right
   - Select "New repository"
   - Repository name: `Re3-ReLU-Region-Reason` (or your preferred name)
   - Description: "Interactive tool for interpreting ReLU neural networks through piecewise-affine mapping"
   - Choose **Public** (for open source) or **Private** (for private research)
   - **DO NOT** initialize with README, .gitignore, or license (we'll add these)
   - Click "Create repository"

### Step 3: Prepare Files for Initial Commit

Before committing, ensure your `.gitignore` file is in place (already created in the project root):

```bash
# Check that .gitignore exists
ls -la .gitignore

# Review what will be excluded
cat .gitignore
```

**Important**: The `.gitignore` file excludes:
- Trained model files (`*.pth`, `*.pt`) - these are large and should not be in git
- Python cache files, virtual environments, IDE-specific files
- Jupyter notebooks (`*.ipynb`), `seeds_dataset.txt`, `spambase.*` (only Project Structure + CSVs are tracked)
- `.env` (never commit API keys)

#### Uploading Only Project Structure Files and CSVs

If you want to push **only** the files listed in the [Project Structure](README.md#-project-structure) in README.md **plus** your CSV dataset files, use the following.

**Files to include:**

| Category | Files |
|----------|--------|
| **Project structure** | `app.py`, `re3_core.py`, `quick_start.py`, `requirments.txt`, `.gitignore`, `README.md`, `APP_README.md`, `QUICK_START_GUIDE.md`, `CONTRIBUTING.md` |
| **Optional** | `.env.example` (template for API key; no secrets) |
| **Datasets (CSVs)** | `iris_training_set.csv`, `iris_test_set.csv`, `accelerometer_Training_set.csv`, `accelerometer_Test_set.csv`, `diabetes_Training_set.csv`, `diabetes_test_set.csv` (or any other `.csv` you use) |

**Method 1 – Add only these files (recommended):**

```bash
# From your project root (e.g. Re3-working_rd)

# 1. Add Project Structure files
git add app.py re3_core.py quick_start.py requirments.txt .gitignore
git add README.md APP_README.md QUICK_START_GUIDE.md CONTRIBUTING.md

# 2. Optional: add .env.example (safe; no secrets)
git add .env.example

# 3. Add CSV datasets
git add *.csv

# 4. Check what will be committed
git status

# 5. Commit
git commit -m "Initial commit: Re3 app (Project Structure + CSVs)"
```

**Method 2 – Use .gitignore and add everything else:**

The repo’s `.gitignore` is set up so that `git add .` will **not** add:
- `*.pth`, `*.pt`, `.env`, notebooks, `seeds_dataset.txt`, `spambase.*`, cache, venv, etc.

So you can also do:

```bash
git add .
git status   # Confirm only Project Structure files + CSVs are staged
git commit -m "Initial commit: Re3 app (Project Structure + CSVs)"
```

Only the Project Structure files, `.env.example` (if present), and CSV files will be staged; everything else is ignored.

### Step 4: Initial Commit and Push

```bash
# If you used Method 1 above, you already ran git add and git commit.
# If you use Method 2, stage and commit:
# Stage all files (except those in .gitignore)
git add .

# Check what will be committed
git status

# Create initial commit
git commit -m "Initial commit: Re3 Interactive Application

- Streamlit web application for ReLU network interpretability
- Support for 1-3 hidden layer networks
- Single sample and region analysis capabilities
- Support for multiple datasets (Iris, Seeds, Spambase, Accelerometer, Diabetes)
- Interactive model training script
- Comprehensive documentation"

# Add remote repository (replace with your GitHub username and repo name)
git remote add origin https://github.com/YOUR_USERNAME/Re3-ReLU-Region-Reason.git

# Verify remote was added
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

**Note**: If you're using SSH instead of HTTPS:
```bash
git remote add origin git@github.com:YOUR_USERNAME/Re3-ReLU-Region-Reason.git
```

### Step 5: Regular Updates and Pushing Changes

After making changes to your code, follow this workflow:

#### 1. Check Status
```bash
# See what files have changed
git status

# See detailed changes
git diff
```

#### 2. Stage Changes
```bash
# Stage specific files
git add app.py
git add re3_core.py
git add quick_start.py

# Or stage all changes
git add .
```

#### 3. Commit Changes
```bash
# Commit with descriptive message
git commit -m "Add About tab with project documentation

- Added comprehensive About tab to Streamlit app
- Included project description, target audience, and benefits
- Updated documentation with technical foundation details"

# Or use a more detailed multi-line message
git commit -m "Fix IndexError in class name indexing

- Added bounds checking for class_names array access
- Fixed TypeError for numpy float64 to int conversion
- Improved error handling in prepare_data function
- Updated analyze_regions to handle edge cases"
```

#### 4. Push to GitHub
```bash
# Push changes to main branch
git push origin main

# If this is your first push to a new branch
git push -u origin main
```

### Step 6: Creating Meaningful Commit Messages

Follow these guidelines for commit messages:

**Good Commit Messages:**
```
✅ "Add support for Diabetes dataset in quick_start.py"
✅ "Fix IndexError when accessing class_names with sparse indices"
✅ "Update documentation with GitHub publishing instructions"
✅ "Improve error handling for non-numeric column detection"
```

**Bad Commit Messages:**
```
❌ "update"
❌ "fix bug"
❌ "changes"
❌ "asdf"
```

**Format:**
- First line: Brief summary (50 characters or less)
- Blank line
- Detailed explanation (if needed)
- Use imperative mood ("Add feature" not "Added feature")

### Step 7: Working with Branches (Optional but Recommended)

For larger features or experiments, use branches:

```bash
# Create a new branch
git checkout -b feature/about-tab

# Make your changes, then commit
git add .
git commit -m "Add About tab to application"

# Push branch to GitHub
git push -u origin feature/about-tab

# Switch back to main branch
git checkout main

# Merge feature branch (after testing)
git merge feature/about-tab

# Delete local branch
git branch -d feature/about-tab

# Delete remote branch
git push origin --delete feature/about-tab
```

### Step 8: Tagging Releases

For important milestones or versions:

```bash
# Create a tag for version 1.0
git tag -a v1.0 -m "Release version 1.0

- Full Streamlit application with all features
- Support for multiple datasets
- Comprehensive documentation
- About tab with project information"

# Push tags to GitHub
git push origin v1.0

# Or push all tags
git push origin --tags
```

### Step 9: Updating Documentation

When updating the project, also update documentation:

```bash
# After making changes to code
git add app.py re3_core.py quick_start.py
git add APP_README.md QUICK_START_GUIDE.md
git commit -m "Update app and documentation

- Added new feature X
- Updated README with new instructions
- Fixed documentation typos"
git push origin main
```

### Step 10: Collaborating with Others

If others will contribute:

1. **Create CONTRIBUTING.md** (optional but recommended):
   ```bash
   # Create contributing guidelines
   touch CONTRIBUTING.md
   ```

2. **Add Collaborators** (on GitHub):
   - Go to repository → Settings → Collaborators
   - Add collaborators by username or email

3. **Pull Latest Changes** (before starting work):
   ```bash
   git pull origin main
   ```

4. **Resolve Conflicts** (if any):
   ```bash
   # If conflicts occur during pull
   git status  # See conflicted files
   # Edit files to resolve conflicts
   git add .
   git commit -m "Resolve merge conflicts"
   git push origin main
   ```

### Common Git Commands Reference

```bash
# View commit history
git log
git log --oneline  # Compact view

# View changes in a specific file
git diff app.py

# Undo changes (before staging)
git checkout -- app.py

# Unstage a file
git reset HEAD app.py

# View remote repositories
git remote -v

# Update remote URL (if needed)
git remote set-url origin NEW_URL

# Clone repository (for others)
git clone https://github.com/YOUR_USERNAME/Re3-ReLU-Region-Reason.git

# Fetch latest changes without merging
git fetch origin

# See differences between local and remote
git diff main origin/main
```

### Best Practices

1. **Commit Frequently**: Make small, logical commits rather than large ones
2. **Write Clear Messages**: Describe what and why, not just what
3. **Pull Before Push**: Always pull latest changes before pushing
4. **Test Before Commit**: Ensure code works before committing
5. **Don't Commit Sensitive Data**: Never commit API keys, passwords, or personal data
6. **Use .gitignore**: Keep large files and temporary files out of git
7. **Review Changes**: Use `git status` and `git diff` before committing

### Troubleshooting Git Issues

**Problem**: "Updates were rejected because the remote contains work"
```bash
# Solution: Pull first, then push
git pull origin main
git push origin main
```

**Problem**: "Large file detected" (GitHub has 100MB file limit)
```bash
# Solution: Add file to .gitignore and remove from git
git rm --cached large_file.pth
git commit -m "Remove large file from tracking"
```

**Problem**: "Authentication failed"
```bash
# Solution: Use personal access token or SSH keys
# For HTTPS: Generate token at GitHub Settings → Developer settings
# For SSH: Set up SSH keys in GitHub Settings → SSH and GPG keys
```

### GitHub Repository Settings

After publishing, configure:

1. **Repository Description**: Add clear description
2. **Topics/Tags**: Add relevant tags (e.g., `machine-learning`, `interpretability`, `pytorch`, `streamlit`)
3. **Website**: Add link if you deploy the app
4. **License**: Add appropriate license file (e.g., MIT, Apache 2.0)
5. **README**: Ensure README.md is comprehensive (already exists)

### Next Steps After Publishing

1. **Add Badges** (optional): Add status badges to README.md
2. **Create Issues Template**: For bug reports and feature requests
3. **Set Up Actions** (optional): For CI/CD automation
4. **Deploy App** (optional): Deploy Streamlit app using Streamlit Cloud or other platforms

---

**Congratulations!** Your Re3 project is now on GitHub and ready for collaboration and version control! 🎉

