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
cd ntu-ktp-data-quality/DuplicateDetectionTool
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
---

### Example Usage
1. Upload your dataset (e.g., ```test.csv```) to the app.

2. Detect and classify:
   - Duplicate records
   - Overly similar records
   - Unique records

3. Visualize results interactively.
   
4. Export results as:
```duplicates.csv```
```overly_similar.csv```
```unique_records.csv```

---


### **Test Dataset Description**

The test dataset is included to demonstrate the capabilities of the **Duplicate Detection Tool**. It simulates common scenarios in data quality management, including duplicates, similar records, and unique entries. Below is an overview of the dataset:

#### **Key Features**
- **File Name:** `test.csv`
- **Format:** CSV file
- **Purpose:** To test and showcase the tool's ability to:
  - Detect duplicate records
  - Identify overly similar records
  - Distinguish unique entries
- **Size:** Small-scale dataset for quick testing and demonstration.

---

#### **Dataset Columns**
| **Column Name**       | **Description**                                          | **Example**            |
|------------------------|----------------------------------------------------------|------------------------|
| `ID`                  | Unique identifier for each record.                       | `001`, `002`, `003`   |
| `Name`                | Name of an individual or entity.                         | `John Doe`            |
| `Email`               | Email address for communication.                        | `john.doe@example.com`|
| `Phone`               | Contact number in various formats.                      | `+1-123-456-7890`     |
| `Address`             | Residential or business address.                        | `123 Main Street`     |
| `Date`                | Date of registration or activity (ISO 8601 format).      | `2023-12-25`          |
| `Other Attributes`    | Additional columns with categorical or numerical data.  | `Category A`, `1000`  |

---

#### **Usage Instructions**
1. Place the test dataset (`test.csv`) in the appropriate directory.
2. Run the tool to detect duplicates, overly similar records, and unique records:
   ```bash
   python main.py test.csv

---


### Images 📸
Landing Page After Data Upload
![Screenshot of Landing Page After Data Upload](Screenshots/Screenshot%202025-01-10%20at%2014.10.30.png)

---


Duplicate Detection Interface
![Screenshot of Duplicate Detection Tool](Screenshots/Screenshot%202025-01-10%20at%2014.10.53.png)

---



Results Visualization
![Screenshot of Results Visualization](Screenshots/Screenshot%202025-01-10%20at%2014.12.41.png)

---


Export Options
![Screenshot of Export Options](Screenshots/Screenshot%202025-01-10%20at%2014.12.49.png)

---


### Contributing
Contributions are welcome! Feel free to:

Open issues for bug reports or feature requests.
Submit pull requests for improvements or fixes.

### License
This project is licensed under the MIT License. See the LICENSE file for details.



