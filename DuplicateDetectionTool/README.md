# Duplicate Detection Tool 🔍

This tool identifies **duplicate**, **overly similar**, and **unique records** in datasets using a combination of **Non-ML** and **ML-based techniques**. It includes:
- **Command Line Interface (CLI):** For batch processing.
- **Streamlit App:** For interactive detection and visualization.

---

## Features

- 🔗 **Hybrid Detection:** Combines Non-ML and ML approaches, prioritizing Non-ML methods for enhanced accuracy.
- 📊 **Interactive App:** Visualize results and inspect duplicates, similar records, and unique entries.
- 📂 **Export Results:** Save results as CSV files for further analysis.
- ⚙️ **Customizable Thresholds:** Adjust thresholds for similarity and duplication as needed.

---

## Installation

### 1. Clone the Repository

```
git clone https://github.com/itsbigspark/ntu-ktp-data-quality.git
cd ntu-ktp-data-quality/DuplicateDetectionTool/KTP_Dupes
```

### 2. Set Up the Environment
It is recommended to use a virtual environment to manage dependencies:
```
python -m venv venv
```

Activate the environment:

On macOS/Linux:
```
source venv/bin/activate
```

On Windows:
```
venv\Scripts\activate
```

Install dependencies:
```
pip install -r requirements.txt
```


### 3. Run the Tool

To process your dataset via the CLI:
```
python main.py <your-dataset.csv>
```

To use the Streamlit App, follow these steps:
```
cd duplicate_detection_tool
streamlit run duplicates_app.py
```

### Example Usage
1. Upload your dataset (e.g., ```test.csv```) to the app.

2. Detect and classify:
*Duplicate records
*Overly similar records
*Unique records

3. Visualize results interactively.
   
4. Export results as:
```duplicates.csv```
```overly_similar.csv```
```unique_records.csv```


Images 📸\
Landing Page After Data Upload


Duplicate Detection Interface


Results Visualization


Export Options


Contributing
Contributions are welcome! Feel free to:

Open issues for bug reports or feature requests.
Submit pull requests for improvements or fixes.\

License
This project is licensed under the MIT License. See the LICENSE file for details.


