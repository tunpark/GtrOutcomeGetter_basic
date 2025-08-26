# -GtrOutcomeGetter
This repository contains the training and testing code for the GtrOutcomeGetter API, a backend service hosted on Hugging Face Spaces that powers the "GtR Project Analyzer" browser extension.
1.  **Text Classification**: To determine whether a project outcome's description pertains to "software".
2.  **Named Entity Recognition (NER)**: To extract specific entities like software names and support URLs from the text.

The repository is structured into a clear, step-by-step pipeline.

## Data Availability

Due to the nature of the data source (fetching from a live API), this repository cannot directly provide the raw or complete datasets.

However, the repository provides some processed annotated dataset samples for demonstrating and testing the functionality of the entire code pipeline. Please note the following descriptions of these samples:

* **`cleaned_all.jsonl`**: This is the dataset used for the NER task. It is a cleaned file derived from annotations made in Doccano. This dataset was annotated by a single person in a single pass. Therefore, its reference value for academic or production purposes is limited.

* **`combined_analysis_data.csv`**: This is the dataset used for the text classification task. It is a composite dataset created as follows: first, a weak-label set was generated using weak-supervision rules. This set was then manually corrected over 64 iterations in Label Sleuth to create a "mixed-label set". Finally, this mixed-label set was combined with a high-quality "golden dataset". Samples in the final file are marked with `weak` (representing the mixed-label set) and `gold` sources. This structure is specifically designed for performing weight testing in subsequent model training. This dataset is also for reference only.

For best results, users are strongly encouraged to run the scripts in **Stage 1** to generate their own up-to-date and complete dataset.

## Project Workflow

The entire project follows a logical sequence, organized into folders. Follow the steps in order to replicate the analysis.

### Stage 1: Data Acquisition and Preparation (`getData/`)

This stage focuses on gathering raw data from the API and processing it through a series of rule-based and manual steps to create the final training datasets.

* **1.1. Data Acquisition (`sample_projects.py` & `fetch_outcomes.py`)**:
    * These scripts fetch project data from the UKRI GtR API based on funder quotas.
    * **Output**: A SQLite database (`projects_sample.db`) containing raw project and outcome text.

* **1.2. Rule-Based Dataset Generation (`getGodlendata.py` & `getWeakData.py`)**:
    * These scripts create initial labeled datasets for the **classification task** from the database.
    * `getGodlendata.py`: Applies hard-coded rules to create a high-quality, "golden" dataset. This dataset also serves as a foundation for guiding the manual NER annotation process.
        * **Output**: `software_and_creative_outcomes.csv`.
    * `getWeakData.py`: Uses keyword matching to generate a larger, "weakly" labeled dataset.
        * **Output**: `weakly_labeled_outcomes.csv`.

* **1.3. Cleaning (NER Workflow)**:
    * The golden datasets generated above, are used to put in a tool like **Doccano**. The goal is to create a detailed dataset for the NER task.
    * `getCleanDoccano.py`: This script cleans the raw `all.jsonl` file from Doccano. It analyzes the data balance and intelligently removes samples with no entities if necessary.
        * **Input**: `all.jsonl`
        * **Output**: `cleaned_all.jsonl`.

* **1.4. Final Data Formatting & Combination**:
    * `toBIO.py`: Converts the cleaned NER data (`cleaned_all.jsonl`) into the BIO format, which is required for some NER models.
        * **Input**: `cleaned_all.jsonl`
        * **Output**: `bio_format.txt`.
    * `getTestData.py`: Formats the golden classification dataset for specific tools.
        * **Input**: `software_and_creative_outcomes.csv`
        * **Output**: `labelsleuth.csv`.
    * **(Combination for Classification)** The golden and weak classification datasets are combined into a single file for training. A helper function for this is located in `model_training/classifier_modelchoice.py`.
        * **Output**: `combined_analysis_data.csv`.

### Stage 2: Model Training & Experimentation (`model_training/`)

This stage involves experimenting with different models and techniques to find the best-performing one for each task.

#### Part A - Text Classification

* **2.1. `classifier_modelchoice.py`**:
    * **Purpose**: To find the best baseline model and the optimal weight for weak labels.
    * **Process**: Tests multiple traditional classifiers across a range of `sample_weight` values.
    * **Output**: Plots and a CSV report (`weight_analysis_report.csv`).

* **2.2. `classifier_oversampling.py`**:
    * **Purpose**: To address class imbalance using oversampling techniques.
    * **Process**: Takes the best weight from the previous step and tests various oversampling strategies (e.g., SMOTE).
    * **Output**: Plots and a CSV report (`imbalanced_sampling_results.csv`).

* **2.3. `save_classifier.py`**:
    * **Purpose**: To train and save the final, production-ready model.
    * **Process**: Takes the best-performing configuration (e.g., Logistic Regression + SMOTE), trains it, and saves the final model artifacts into the `final_model/` directory.

#### Part B - Named Entity Recognition (NER)

* **`train_spacy.py` / `train_transformer.py` / `train_bilstm.py`**:
    * **Purpose**: Each script trains a NER model using a different popular framework, allowing for a comparison of approaches.
    * **Process**: They load the prepared data, handle balancing, train the model, and save the final artifacts.

### Stage 3: In-depth Model Evaluation (`model_evaluating/`)

This stage is for conducting a deep dive into the performance of the final, saved models to understand their strengths and weaknesses.

* **3.1. `error_analysis_classifier.py`**:
    * **Purpose**: Performs a detailed error analysis on the **saved text classifier**.
    * **Process**: It loads the final model, regenerates the test set, and produces detailed reports and visualizations, including a confusion matrix, ROC curve, and CSV files listing False Negatives (FN) and False Positives (FP).

* **3.2. `error_analysis_nre.py`**:
    * **Purpose**: Performs a span-based error analysis on the **saved NER model**.
    * **Process**: It loads a saved NER model and classifies errors into categories like `boundary_error` and `class_error`.
    * **Output**: Detailed CSV reports and plots.

### Stage 4: Model Inference and Usage

This final stage demonstrates how to load and use the saved models for prediction on new data.

* **Prediction Functions**: The `save_classifier.py` script contains helper functions (`load_model_and_predict`) that serve as a template for loading the final model and making predictions on new text.

## How to Run the Pipeline

### Important Note on Configuration

**After downloading this repository, you will need to manually adjust the file paths inside the scripts.** Many scripts contain hardcoded paths to data files, models, or output directories (e.g., in `if __name__ == "__main__":` blocks). Please review the scripts you intend to run and update these paths to match your local file structure.

1.  **Setup**:
    * Install all required Python packages.
    * 
2.  **Data Preparation**:
    * Run the scripts in the `getData/` folder in the logical order described in Stage 1 to fetch and process the data.

3.  **Model Training**:
    * For classification, run the scripts in `model_training/` in order:
        1.  `classifier_modelchoice.py`
        2.  `classifier_oversampling.py`
        3.  `save_classifier.py`
    * For NER, choose one of the training scripts (e.g., `train_transformer.py`) and run it.

4.  **Model Evaluation**:
    * After a final model has been saved, run the corresponding script in the `model_evaluating/` folder to get a detailed performance breakdown.






