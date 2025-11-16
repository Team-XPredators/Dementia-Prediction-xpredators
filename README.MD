# XPredators Dementia Prediction

This project implements a machine learning pipeline for dementia prediction using a dataset from Google Drive. It includes data loading, preprocessing, feature selection, dimensionality reduction, and training of classification models (Logistic Regression and Random Forest) to predict binary dementia outcomes.

## Key Features

- **Data Loading and Exploration**: Downloads and loads a CSV dataset with 1024 columns and approximately 195k rows, containing mixed numerical and object data types.
- **Data Preprocessing**: Removes medical-related and object columns, handles missing values by filling with medians, and applies variance thresholding to eliminate low-variance features.
- **Feature Selection and Reduction**: Uses PCA to reduce features from 645 selected to 82 principal components while retaining 95% variance.
- **Model Training and Evaluation**: Trains Logistic Regression and Random Forest models on an 80/20 train/test split, providing accuracy, precision, recall, F1-score, and confusion matrices.

## Tech Stack

- **Languages**: Python
- **Libraries**:
  - `gdown` for downloading files from Google Drive
  - `pandas` for data manipulation
  - `scikit-learn` for preprocessing, feature selection (VarianceThreshold), dimensionality reduction (PCA), model training (LogisticRegression, RandomForestClassifier), and evaluation metrics
- **Environment**: Jupyter Notebook (e.g., Google Colab or local Jupyter)

## Project Structure

```
.
├── XPredators_Demantia_Prediction.ipynb  # Main notebook containing the full ML pipeline
└── .ipynb_checkpoints/                   # Jupyter checkpoint directory for version control
    └── Untitled-checkpoint.ipynb         # Automatic backup of an empty notebook state
```

## Setup Instructions

1. Ensure Python 3.x is installed.
2. Install required dependencies:
   ```bash
   pip install gdown pandas scikit-learn
   ```
3. Set up a Jupyter environment, such as Jupyter Notebook or Google Colab.
4. Download the project files or clone the repository containing the notebook.

## Usage

1. Open `XPredators_Demantia_Prediction.ipynb` in Jupyter Notebook or Colab.
2. Run cells sequentially:
   - Run pip installs and import statements.
   - Execute data download and loading from Google Drive.
   - Perform data exploration, cleaning, and preprocessing.
   - Apply feature selection and PCA reduction.
   - Train and evaluate the Logistic Regression and Random Forest models.
3. Review outputs, including classification reports (e.g., accuracy around 70-80% depending on configuration).
4. Adjust hyperparameters (e.g., Random Forest n_estimators=300, PCA variance=0.95) as needed for tuning.

## Contributing

Contributions are welcome. Please submit issues or pull requests for improvements to the pipeline, models, or documentation.

## License

This project is licensed under the MIT License. See LICENSE for details.