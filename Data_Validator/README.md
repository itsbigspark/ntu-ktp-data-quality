# **Data Validator Tool 🔧**

This tool provides a robust solution for **data validation**, **cleaning**, and **visualization**. It identifies **missing**, **invalid**, and **incorrect patterns** in datasets  It includes:
- **Command Line Interface (CLI):** For batch processing.
- **Streamlit App:** For interactive detection and visualization.

---

## Features

Features
- ✅ Data Validation: Detects missing and invalid values using custom or inferred rules.
- 🧹 Data Cleaning: Automatically corrects invalid values using inferred patterns and predefined rules.
- 📊 Metrics Calculation: Computes AVH metrics (Accuracy, Validity, and Homogeneity) to assess dataset quality.
- ⚙️ Custom Rules: Accepts user-defined JSON rules for validation.
- 🤖 Synthetic Data Generation: Creates synthetic datasets for testing and validation purposes.
- 🌟 Interactive Streamlit App: Upload datasets, process validation, and download results with a user-friendly UI.
- 📈 Visualizations: Provides insights into issue distribution and AVH metrics through graphical representations.

---

## Installation

### 1. Clone the Repository

```
git clone https://github.com/itsbigspark/ntu-ktp-data-quality.git
cd ntu-ktp-data-quality/Data_Validator
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


## 3. Run the Tool

### To process your dataset via the CLI:
```
python main.py <path_to_clean_data.csv> <path_to_unclean_data.csv> [--rules <path_to_custom_rules.json>]

```

To use the Streamlit App, follow these steps:
```
cd duplicate_detection_tool
streamlit run duplicates_app.py
```

### Arguments:

- <path_to_clean_data.csv>: Path to the clean dataset.
- <path_to_unclean_data.csv>: Path to the unclean dataset.
- [--rules <path_to_custom_rules.json>]: Optional JSON file for custom rules.

  
---

### Example Usage

```
python main.py examples/cleaned_data.csv examples/unclean_data.csv
```

---

### Streamlit App
Launch the Streamlit app:
``` 
streamlit run appy.py
```

### Steps:

- Upload the **Error Data** File (required).
- Upload the **Reference Data** File (optional).
- Upload **Custom Rules** (optional).
- Click **Start Validation and Cleaning**.

### View and download:

- **Cleaned Data**
- **Highlighted Issues**
- **Synthetic Data**
- **Visualizations**

--

### Output Files
1. **Cleaned Data** (```cleaned_data.csv```): A cleaned version of your input dataset.
2. **Highlighted Issues** (```highlighted_issues.xlsx```): An Excel file with color-coded issues:
  - Yellow for **MISSING** values.
  - Red for **INVALID** values.
3. **Metrics File** (```metrics.json```): A JSON file with computed patterns and AVH metrics.
4. **Visualizations**:
  - Issue distribution plot: ```outputs/visualizations/issue_distribution.png```
  - AVH metrics plot: ```outputs/visualizations/metrics_plot.png```


### **Example Dataset Description**

TThe example datasets provided in the ```Examples/``` directory demonstrate common data quality issues, including missing and invalid values. Below is an overview of the dataset:



#### **Dataset Columns**

| **Column Name**         | **Description**                                          | **Example**             |
|--------------------------|----------------------------------------------------------|-------------------------|
| `CompanyName`           | Name of the company                                      | `XYZ Ltd.`              |
| `CompanyNumber`         | Unique company identifier                                | `12345678`              |
| `RegAddress_CareOf`     | Care of address line (optional)                          | `John Doe`              |
| `RegAddress_POBox`      | PO Box number (optional)                                 | `PO Box 123`            |
| `RegAddress_Address1`   | Primary address line                                     | `123 Main Street`       |
| `RegAddress_Address2`   | Secondary address line                                   | `Suite 456`             |
| `RegAddress_PostTown`   | Town or city of the registered address                   | `London`                |
| `RegAddress_County`     | County or region of the registered address               | `Greater London`        |
| `RegAddress_PostCode`   | Postal code for the registered address                   | `AB12 3CD`             |
| `CountryOfOrigin`       | Country where the company is registered                  | `United Kingdom`        |
| `Date`                  | Registration date in ISO 8601 format                     | `2023-12-25`            |


---

## Visualization Examples

### Landing Page After Data Upload

### Cleaned Data Output

### Highlighted Issues Visualization

--


## Customization

### Custom Rules
Custom rules can be provided in JSON format. Example:

```
{
  "patterns": {
    "CompanyNumber": "^\\d+$",
    "RegAddress_POBox": "^\\d{1,6}$"
  },
  "metrics": {
    "CompanyNumber": {"valid_percentage": 95},
    "RegAddress_POBox": {"valid_percentage": 80}
  }
}
```

--

## Extend Functionality

### To add new features:

- Modify the scripts in the data_validator/ directory.
- For cleaning: Update cleaning.py.
- For metrics: Add logic to metrics.py.
- For visualization: Extend visualization.py.

--
## Testing

### Run automated tests to ensure functionality:
```
python -m unittest discover tests
```

## Test Files:

```tests/test_cleaning.py```: Tests for cleaning logic.
```tests/test_metrics.py```: Tests for metrics calculation.
```tests/test_patterns.py```: Tests for regex pattern inference.

--

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.

2. Create a feature branch:
```git checkout -b feature-name```

3. Commit your changes:
```git commit -m "Added new feature"```

4. Push the branch:
```git push origin feature-name```

5. Submit a pull request.


---


## Images 📸
### Landing Page
![Screenshot of Landing Page](Data_Validator/images/Screenshot%2025-01-16%at%14.01.29.png)

---


### Data Upload
![Screenshot of Data Upload](Data_Validator/images/Screenshot%2025-01-16%at%14.02.02.png)

---



### Inferring Patterns
![Screenshot of Pattern Inference](Data_Validator/images/Screenshot%2025-01-16%at%14.03.00.png)

---
Data_Validator/images/Screenshot 2025-01-16 at 14.01.29.png

### Cleaning Rules
![Screenshot of  Cleaning Rules](Data_Validator/images/Screenshot%2025-01-16%at%14.03.37.png)

---



## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or support:

Email: preetham.yuvaraju@bigspark.dev
GitHub Issues: Submit an Issue



