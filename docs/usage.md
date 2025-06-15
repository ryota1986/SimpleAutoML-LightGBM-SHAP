# Usage Guide

This document describes how to run the Streamlit interface and what each option does.

1. Install the dependencies using `pip install -r requirements.txt`.
2. Start the application:

```bash
streamlit run AutoML0.py
```

3. Upload a CSV file. Rows are limited to the first 5,000 records.
4. Select the target column and configure whether you want to enable the time series mode.
5. Click the training button to fit a LightGBM model and view the metrics and feature importance plots.
6. If you supply an OpenAI API key, the app can generate an explanation of the SHAP output.

You can download the trained model (`.pkl`) and the tuned hyperparameters (`.json`) for later use.
