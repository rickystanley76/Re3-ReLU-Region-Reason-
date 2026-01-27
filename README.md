# Re3: ReLU Region Reason - Implementation Notebooks

Jupyter notebooks demonstrating the Re3 method for interpreting ReLU neural networks through piecewise-affine mapping.

## Overview

Re3 (ReLU Region Reason) provides exact, deterministic explanations for neural network predictions by exploiting the piecewise-affine structure of ReLU networks. These notebooks demonstrate the method on the Iris dataset with networks of varying depth.

## Files

### Notebooks
- `re3_1_layer.ipynb` - Re3 analysis on 1-hidden-layer network
- `re3_2_layers.ipynb` - Re3 analysis on 2-hidden-layer network
- `re3_3_layers.ipynb` - Re3 analysis on 3-hidden-layer network

### Datasets
- `iris` - Iris dataset (used in notebooks)
- `seeds_dataset` - Seeds dataset
- `accelerometer_gyro_mobile_phone_dataset` - AGMP dataset
- `diabetes_binary_health_indicators_BRFSS2015` - CDCDHI dataset
- `spambase.data` - Spambase dataset

### Documentation
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Important Note

**The provided notebooks are configured for the Iris dataset only.** To use Re3 with other datasets (Seeds, AGMP, CDCDHI, Spambase), you will need to modify the data loading section in the notebooks to match the specific dataset's structure (number of features, classes, etc.).


## Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy torch scikit-learn scipy lime shap matplotlib jupyter
```

## Usage

### Quick Start (Iris Dataset)

1. Install dependencies (see above)

2. Launch Jupyter:
```bash
   jupyter notebook
```

3. Open any notebook:
   - `re3_1_layer.ipynb` for 1 hidden layer example
   - `re3_2_layers.ipynb` for 2 hidden layers example
   - `re3_3_layers.ipynb` for 3 hidden layers example

4. Run all cells to see complete Re3 analysis

### Using Other Datasets

To adapt the notebooks for other datasets:

1. Modify the data loading section:
```python
   # Change from:
   data = pd.read_csv('iris.csv')
```

2. Update dataset-specific parameters:
   - Number of input features
   - Number of output classes
   - Feature names
   - Class names

3. Adjust network architecture if needed based on dataset complexity

## What the Notebooks Demonstrate

- Model training on Iris dataset
- Region identification and analysis
- Visualization of results

## Requirements

- pandas >= 1.3.0
- numpy >= 1.21.0
- PyTorch >= 1.9.0
- scikit-learn >= 0.24.0
- scipy >= 1.7.0
- lime >= 0.2.0
- shap >= 0.40.0
- matplotlib >= 3.4.0
- jupyter >= 1.0.0

## License

This code is provided for academic and research purposes. If you use this code in your work, please cite the paper below.

## Citation

If you use this code in your research, please cite:
```bibtex
@article{barua2025re3,
  title={Mechanistic Interpretability of ReLU Neural Networks Through Piecewise-Affine Mapping},
  author={Barua, Arnab and Ahmed, Mobyen Uddin and Begum, Shahina},
  journal={Machine Learning},
  year={2025},
  note={Under review}
}
```

## Contact

For questions or issues:
- Email: arnab.barua@mdu.se
- Institution: Mälardalen University, Sweden

## Acknowledgments

This work was conducted at the School of Innovation, Design and Engineering, Mälardalen University, Sweden.

## Related Publication

Barua, A., Ahmed, M. U., & Begum, S. (2025). Mechanistic Interpretability of ReLU Neural Networks Through Piecewise-Affine Mapping. *Machine Learning* (under review).

DOI: [Will add Upon Publishing]