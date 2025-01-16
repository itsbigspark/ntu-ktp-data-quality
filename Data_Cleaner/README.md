# **Data Cleaner Tool 🧹**

This tool provides a robust solution for The Data Cleaner tool provides **automated error detection** and **correction** for datasets, focusing on:

- Resolving **missing values**
- Correcting **typographical errors**
- Handling **out-of-range values**

This tool offers:

1. **Command Line Interface (CLI)** for batch processing.
2. **Streamlit App** for interactive usage and visualization.

---

## Features

- 🔍 Error Detection: Identifies missing, incorrect, or inconsistent data in numerical and categorical columns.
- 🛠️ Error Correction: Corrects detected errors using reference data or statistical methods.
- 📊 Metrics Calculation: Evaluates error detection and correction performance.
- 📂 Export Results: Generates corrected datasets and error-highlighted Excel files.
- 🎨 Interactive App: Upload data, preview corrections, and download results in the Streamlit-based web app.

---

## Installation

### 1. Clone the Repository

```
git clone https://github.com/itsbigspark/ntu-ktp-data-quality.git
cd ntu-ktp-data-quality/Data_Cleaner
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
python main.py <error_data.csv> [reference_data.csv]

```

### Outputs:

```outputs/corrected_data.csv```: Corrected dataset.

```outputs/error_highlighted.xlsx```: Error-highlighted Excel file.

```outputs/corrected_highlighted.xlsx```: Correction-highlighted Excel file.

```outputs/error_detection_metrics.png```: Classification metrics as an image.


### To use the Streamlit App, follow these steps:
```
streamlit run data_cleaner_app.py
```

Features in the App:

1. Upload the Error Data File and optionally the Reference Data File.
2. Preview uploaded datasets.
3. Automatically detect and correct errors.
4. Download:
   - Corrected dataset (corrected_data.csv)
   - Error-highlighted Excel file (error_highlighted.xlsx)
   - Correction-highlighted Excel file (corrected_highlighted.xlsx).
5. View error detection metrics in JSON and bar chart formats.


---

### Example Dataset

| **Column Name**       | **Description**                | **Example**       |
|------------------------|--------------------------------|-------------------|
| `CompanyName`         | Name of the company           | `XYZ Ltd.`       |
| `CompanyNumber`       | Unique company identifier     | `12345678`       |
| `RegAddress_CareOf`   | Registered address line       | `John Doe`       |
| `RegAddress_POBox`    | PO Box number (optional)      | `123`            |
| `RegAddress_Address1` | Primary address line          | `123 Main Street`|
| `Date`                | Registration date (ISO 8601)  | `2023-12-25`     |

---

### Outputs

1. **Corrected Dataset*** (```corrected_data.csv```):
A clean version of your dataset, with all errors resolved.

2. **Error-Highlighted File** (```error_highlighted.xlsx```):
Excel file with detected errors highlighted in RED.

3. **Correction-Highlighted File** (```corrected_highlighted.xlsx```):
Excel file with corrections highlighted in GREEN.

4. **Metrics Imag** (```error_detection_metrics.png```):
A PNG file displaying classification metrics like Precision, Recall, and F1-Score.

---

## How It Works

### Error Detection:

- Detects missing values.
- Identifies typographical errors in categorical columns.
- Detects out-of-range values in numerical columns.

### Error Correction:

- Fills missing values using:
- Mean/Median for numerical data.
- Mode for categorical data.

### Corrects typographical errors using:

- Reference data for categorical values.
- Adjusts out-of-range values based on statistical thresholds.

### Metrics Calculation:

- Compares corrected data with reference data (if provided).
- Evaluates detection performance using:
- Precision
- Recall
- F1-Score

### Visualization:

Classification metrics are displayed in the Streamlit App and saved as a PNG image.
  
---

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
![Screenshot of Landing Page](images/Screenshot%202025-01-16%20at%2014.32.17.png)

---


### Data Upload
![Screenshot of Data Upload](images/Screenshot%202025-01-16%20at%2014.32.45.png)

---



### Metrics
![Screenshot of Metrics](images/Screenshot%202025-01-16%20at%2014.33.15.png)

---


### Corrected Data
![Screenshot of Corrected Data](images/Screenshot%202025-01-16%20at%2014.33.49.png)


---



## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or support:


Email: preetham.yuvaraju@bigspark.dev






