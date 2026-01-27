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

This interactive script allows you to:
- Train models on **Iris** dataset → saves `trained_iris_model_full.pth` and `trained_iris_model.pth`
- Train models on **Accelerometer/Gyro** dataset → saves `trained_accelerometer_model_full.pth` and `trained_accelerometer_model.pth`
- Train models on **Diabetes** dataset → saves `trained_diabetes_model_full.pth` and `trained_diabetes_model.pth`
- Train all datasets at once by selecting 'all'

The script will show test accuracy for each trained model.

**Alternative**: You can also train models using the Jupyter notebooks (`re3_1_layer.ipynb`, etc.)

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
2. Choose one of:
   - **Upload Model File**: Upload one of the trained models:
     - `trained_iris_model_full.pth` (for Iris dataset)
     - `trained_accelerometer_model_full.pth` (for Accelerometer/Gyro dataset)
     - `trained_diabetes_model_full.pth` (for Diabetes dataset)
     - Or upload your own model
   - **Create New Model**: Configure architecture manually
3. Model status will show ✅ when loaded

### Loading a Dataset

1. In the sidebar, under "2. Load Dataset"
2. Choose one of:
   - **Upload Dataset**: Upload your CSV/TXT file
   - **Use Built-in**: Select Iris, Seeds, or Spambase
3. If uploading, select the target column
4. Click "Prepare Dataset"

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
   - How many unique regions exist
   - Region purity and accuracy
   - Region size distribution
   - Quality analysis plots

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
   - Select "Upload Model File"
   - Upload `trained_iris_model_full.pth`

2. **Load Dataset**:
   - Select "Use Built-in"
   - Choose "Iris"
   - Click "Load Built-in Dataset"

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
   - Select "Upload Model File"
   - Upload `trained_accelerometer_model_full.pth`

2. **Load Dataset**:
   - Select "Upload Dataset"
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
   - Select "Upload Model File"
   - Upload `trained_diabetes_model_full.pth`

2. **Load Dataset**:
   - Select "Upload Dataset"
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
- Contact: arnab.barua@mdu.se

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
- Python cache files
- Virtual environments
- IDE-specific files
- Large dataset files (uncomment if you don't want to track them)

### Step 4: Initial Commit and Push

```bash
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
git add app.py re3_core.py
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

