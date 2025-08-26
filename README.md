# -GtrOutcomeGetter
This repository contains the training and testing code for the GtrOutcomeGetter API, a backend service hosted on Hugging Face Spaces that powers the "GtR Project Analyzer" browser extension.

1.  **Text Classification**: To determine whether a project outcome's description pertains to "software".
2.  **Named Entity Recognition (NER)**: To extract specific entities like software names and support URLs from the text.

The repository is structured into a clear, step-by-step pipeline.

## Data Availability

Due to the nature of the data source (fetching from a live API), this repository cannot directly provide the raw or complete datasets.

However, a sample annotated dataset is included for demonstration purposes. This allows you to run the model training and evaluation pipeline out-of-the-box to verify its functionality. **Please note that this sample dataset is for demonstration only and may not have significant reference value for academic or production purposes.**

For best results, users are strongly encouraged to run the scripts in **Stage 1** to generate their own up-to-date and complete dataset.

## Project Workflow

The entire project follows a logical sequence, organized into folders. Follow the steps in order to replicate the analysis.

### Stage 1: Data Acquisition and Preparation (`getData/`)

This stage focuses on gathering raw data from the UKRI GtR API, processing it, and creating labeled datasets for model training.

* **1.1. `sample_projects.py` & `fetch_outcomes.py`**:
    * These scripts work together to fetch project data. `sample_projects.py` first samples projects based on funder quotas, and then `fetch_outcomes.py` retrieves the detailed outcome descriptions for those projects.
    * **Output**: A SQLite database (`projects_sample.db` or `project_outcomes.db`) containing raw project and outcome text.

* **1.2. `getCleanDoccano.py` (Crucial Pre-processing)**:
    * This script intelligently cleans the raw JSONL data exported from an annotation tool like Doccano (e.g., `all.jsonl`).
    * **Process**: It first analyzes the distribution of samples with and without entities. Based on the imbalance ratio, it recommends and performs the removal of empty-entity samples to create a more balanced dataset for training.
    * **Output**: `cleaned_all.jsonl`. This cleaned file should be used as the input for subsequent data formatting and training steps.

* **1.3. `getGodlendata.py` & `getWeakData.py`**:
    * These scripts create labeled datasets from the raw data in the database.
    * `getGodlendata.py`: Applies hard-coded rules to create a high-quality, "golden" dataset.
        * **Output**: `software_and_creative_outcomes.csv` (the golden dataset).
    * `getWeakData.py`: Uses keyword matching on URLs to generate a larger but potentially noisy "weakly" labeled dataset.
        * **Output**: `weakly_labeled_outcomes.csv`.

* **1.4. Data Formatting (`getTestData.py`, `toBIO.py`)**:
    * These are utility scripts for converting data into specific formats required by different tools or models.
    * `getTestData.py`: Formats the golden dataset for use in annotation tools like Label Sleuth.
        * **Output**: `labelsleuth.csv`.
    * `toBIO.py`: Converts the cleaned JSONL data (`cleaned_all.jsonl`) into the BIO format for NER models.
        * **Output**: `bio_format.txt`.

* **1.5. Final Data Combination**:
    * Before training, the golden and weak datasets are combined into a single file. A helper function for this is included in `model_training/classifier_modelchoice.py`.
    * **Output**: `combined_analysis_data.csv`. This is the primary input file for the classification models.

### Stage 2: Model Training & Experimentation (`model_training/`)

This stage involves experimenting with different models and techniques to find the best-performing one for each task.

#### Part A - Text Classification

The classification task follows a three-step experimental process to build the best model.

* **2.1. `classifier_modelchoice.py`**:
    * **Purpose**: To find the best baseline model and the optimal weight for weak labels.
    * **Process**: Tests multiple traditional classifiers (Logistic Regression, SVM, Random Forest) across a range of `sample_weight` values for the weak data.
    * **Output**: Plots and a CSV report (`weight_analysis_report.csv`) identifying the best model and weight.

* **2.2. `classifier_oversampling.py`**:
    * **Purpose**: To address class imbalance.
    * **Process**: Takes the best weight from the previous step as a baseline and tests various oversampling techniques (e.g., SMOTE) to see if they further improve performance.
    * **Output**: Plots and a CSV report (`imbalanced_sampling_results.csv`) comparing sampling strategies.

* **2.3. `save_classifier.py`**:
    * **Purpose**: To train and save the final, production-ready model.
    * **Process**: Takes the best-performing configuration (e.g., Logistic Regression + SMOTE), trains it on the data, and saves the final model artifacts (`.joblib` files, metadata) into the `final_model/` directory.

#### Part B - Named Entity Recognition (NER)

This part explores different architectures for the NER task.

* **`train_spacy.py` / `train_transformer.py` / `train_bilstm.py`**:
    * **Purpose**: Each script trains a NER model using a different popular framework.
    * **Process**: They each load the prepared data, handle data balancing, train the model, and save the final artifacts to their respective output directories (e.g., `spacy_ner_model/`, `transformer_ner_model/`).

### Stage 3: In-depth Model Evaluation (`model_evaluating/`)

This stage is for conducting a deep dive into the performance of the final, saved models to understand their strengths and weaknesses.

* **3.1. `error_analysis_classifier.py`**:
    * **Purpose**: Performs a detailed error analysis on the **saved text classifier**.
    * **Process**: It loads the final model from `final_model/`, regenerates the test set, and produces detailed reports and visualizations, including a confusion matrix, ROC curve, and CSV files listing all False Negative (FN) and False Positive (FP) samples.

* **3.2. `error_analysis_nre.py`**:
    * **Purpose**: Performs a span-based error analysis on the **saved NER model**.
    * **Process**: It loads a saved Transformer-based NER model, runs it on the test set, and classifies errors into categories like `boundary_error` (correct label, wrong span) and `class_error` (correct span, wrong label).
    * **Output**: Detailed CSV reports and plots, such as a confusion matrix for class errors.

### Stage 4: Model Inference and Usage

This final stage demonstrates how to load and use the saved models for prediction on new data.

* **Prediction Functions**: The `save_classifier.py` script contains helper functions (`load_model_and_predict`) that serve as a template for loading the final model and making predictions on new text.

## How to Run the Pipeline

### Important Note on Configuration

**After downloading this repository, you will need to manually adjust the file paths inside the scripts.** Many scripts contain hardcoded paths to data files, models, or output directories (e.g., in `if __name__ == "__main__":` blocks). Please review the scripts you intend to run and update these paths to match your local file structure.

1.  **Setup**:
    * Install all required Python packages. It is recommended to create a virtual environment.
        ```bash
        pip install -r requirements.txt
        ```

2.  **Data Preparation**:
    * Run the scripts in the `getData/` folder in the logical order described above to fetch and process the data.
    * Ensure the final `combined_analysis_data.csv` (for classification) and `bio_format.txt` / `cleaned_all.jsonl` (for NER) are generated.

3.  **Model Training**:
    * For classification, run the scripts in `model_training/` in order:
        1.  `classifier_modelchoice.py`
        2.  `classifier_oversampling.py`
        3.  `save_classifier.py`
    * For NER, choose one of the training scripts (e.g., `train_transformer.py`) and run it.

4.  **Model Evaluation**:
    * After a final model has been saved (e.g., in `final_model/` or `transformer_ner_model/`), run the corresponding script in the `model_evaluating/` folder to get a detailed performance breakdown.
        * Run `error_analysis_classifier.py` for the classification model.
        * Run `error_analysis_nre.py` for the NER model.
