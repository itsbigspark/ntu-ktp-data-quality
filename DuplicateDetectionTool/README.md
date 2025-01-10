# Duplicate Detection Tool 🔍

This tool detects duplicate, overly similar, and unique records in datasets using a combination of Non-ML and ML-based techniques. It includes:
- **Command Line Interface (CLI):** For batch processing.
- **Streamlit App:** For interactive detection and visualization.

## Features
- **Hybrid Detection:** Combines Non-ML and ML approaches, with more weight given to Non-ML methods.
- **Interactive App:** Visualize results and inspect duplicates, similar records, and unique records.
- **Export Results:** Save results as CSV files for further analysis.
- **Customizable Thresholds:** Adjust thresholds for similarity and duplication as needed.

## Installation

### 1. Clone the Repository

git clone https://github.com/itsbigspark/ntu-ktp-data-quality.git


cd ntu-ktp-data-quality/DuplicateDetectionTool/KTP_Dupes

### 2. Set Up the Environment
It is recommended to use a virtual environment.

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

### 3. Run the Tool

python main.py <your-dataset.csv>
Streamlit App

streamlit run duplicate_detection_tool/duplicates_app.py



Example
Upload your dataset (e.g., test.csv) and:

Detect duplicate and overly similar records.
View and inspect the results interactively.
Export results as duplicates.csv, overly_similar.csv, and unique_records.csv.


### 4. Directory Structure

DuplicateDetectionTool/

├── duplicate_detection_tool/

│   ├── clustering.py

│   ├── meta_classifier.py

│   ├── preprocessing.py

│   ├── processing.py

│   ├── similarity.py

│   ├── duplicates_app.py

│   └── images/

├── main.py

├── requirements.txt

├── setup.py

├── README.md

├── example_data/


Contributing
Feel free to open issues or submit pull requests for improvements.

License
MIT License
