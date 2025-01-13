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

git clone https://github.com/<your-username>/DuplicateDetectionTool.git
cd DuplicateDetectionTool

### 2. Set Up the Environment
It is recommended to use a virtual environment.

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

### 3. Run the Tool

python main.py <your-dataset.csv>
Streamlit App

streamlit run duplicate_detection_tool/duplicates_app.py