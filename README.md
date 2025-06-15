# SimpleAutoML

SimpleAutoML is a lightweight Streamlit application that automatically trains a LightGBM model from an uploaded CSV file. It supports both regression and classification tasks, provides basic hyperparameter search and SHAP-based model interpretation. When time-series mode is enabled, the app can shift the target column to predict several steps ahead.

## Features

- Automatically detects whether the task is regression or classification
- Allows shifting the target column for time-series prediction
- Hyperparameter tuning via `RandomizedSearchCV`
- SHAP-based model interpretation
- Optional explanation generation using the OpenAI API
- Downloadable trained model and best hyperparameters

## Requirements

Install the required libraries with:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app with:

```bash
streamlit run AutoML0.py
```

Upload your dataset, select the target column and configure any options. If you supply an OpenAI API key, the app can generate a short explanation of the SHAP output.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
