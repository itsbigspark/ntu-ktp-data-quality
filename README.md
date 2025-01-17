# NTU-KTP Data Quality Project 

<p align="center">
  <img src="Resources/bigspark_logo.png" alt="BigSpark Logo" width="200" style="margin: 15px;">
  <img src="Resources/UKRI_logo.png" alt="UKRI Logo" width="200" style="margin: 10px;">
  <img src="Resources/NTU_Primary_logo.png" alt="NTU Logo" width="200" style="margin: 10px;">
</p>


This repository is part of the **Knowledge Transfer Partnership (KTP)** between **Nottingham Trent University (NTU)** and **BigSpark**. The aim of this project is to address **data quality issues** in large datasets specifical;ly in **Finance** using advanced techniques for error detection, error correction, duplicate detection, and beyond.

---

## Project Overview

Data quality is critical for effective decision-making and business operations. The NTU-KTP Data Quality Project is divided into **six stages**, each targeting a specific data quality challenge:

1. **Stage 1: Missing and Incorrect Key Values in the Data**  
   Tools:  
   - **Data Validator**: Detects missing and invalid data based on predefined and inferred patterns.  
   - **Data Cleaner**: Automates error correction for missing, typographical, and out-of-range values.

2. **Stage 2: Duplicate and Overly Similar Records**  
   Tool:  
   - **Duplicate Detection Tool**: Identifies duplicate and overly similar records using a hybrid approach combining **Non-ML** and **ML-based** techniques.

3. **Stage 3: Discrepancies in Attribute Conventions and Entries Identified by the Client**  
   This stage focuses on automated standardization to address discrepancies in data attributes. **Probabilistic record linkage** and **entity resolution models** will be implemented to determine whether two values should belong to the same attribute.  

4. **Stage 4: AI Framework Implementation and Scalability/Maintenance Strategy**  
   This stage aims to unify the individual solutions developed in earlier stages into a cohesive **AI framework** using **model fusion**. Explainable AI (XAI) will be integrated to enhance transparency, and a client software tool will be developed for ease of use.

5. **Stage 5: Real-time Data Ingestion and Quality Testing**  
   This stage involves holistic, end-to-end validation of the AI framework from the perspective of a **data engineer**. It will develop efficient approaches for data quality control stages, tested on fresh datasets.

6. **Stage 6: Embedding of the Knowledge Exchange**  
   Focused on building capacity within the client organization to take ownership of the project. A **Knowledge Base (KB)** will be developed with relevant documentation and guides for using and adapting the tools.

---

## Repository Structure

The repository is organized into the following folders:

| **Folder**                   | **Description**                                                                                      |
|-------------------------------|------------------------------------------------------------------------------------------------------|
| `Data_Validator/`            | Contains tools for validating datasets by identifying missing or invalid values.                    |
| `Data_Cleaner/`              | Includes tools for correcting errors in datasets and generating metrics for error resolution.        |
| `DuplicateDetectionTool/`    | Tools for detecting duplicate and overly similar records in datasets.                               |
| `Error-Identification-Correction/` | Initial implementation of error detection and correction.                                              |
| `AVH-Autovalidate/`          | (Archive) Automated data validation tool for earlier experiments.                                   |
| `README.md`                  | Comprehensive documentation for the repository.                                                     |

---

## Stages and Tools

### Stage 1: Missing and Incorrect Key Values in the Data

#### 1. Data Validator
- **Purpose**: Detects missing, invalid, or incorrectly formatted values in datasets.
- **Features**:  
  - Pattern validation using inferred and custom regex patterns.  
  - Metrics calculation for error analysis.  
  - Visualization of data quality issues.  
- **Implementation**:  
  - CLI-based validation process.  
  - Streamlit app for interactive exploration and visualization.  

#### 2. Data Cleaner
- **Purpose**: Automatically corrects errors in datasets.  
- **Features**:  
  - Correction of missing values.  
  - Typographical error resolution in categorical columns.  
  - Out-of-range value detection and correction for numerical columns.  
- **Implementation**:  
  - CLI-based correction process with Excel exports for highlighted issues.  
  - Streamlit app for interactive correction and result visualization.  

---

### Stage 2: Duplicate and Overly Similar Records

#### Duplicate Detection Tool
- **Purpose**: Identifies duplicate, overly similar, and unique records in datasets.  
- **Features**:  
  - Hybrid detection approach (Non-ML + ML).  
  - Visualization of duplicates and similarities.  
  - Export results as categorized CSV files.  
- **Implementation**:  
  - CLI-based duplicate detection process.  
  - Streamlit app for interactive exploration and threshold adjustments.  

---

### Stage 3: Discrepancies in Attribute Conventions and Entries Identified by the Client

- **Objective**:  
  Standardizes data attributes automatically and resolves discrepancies.  
- **Features**:  
  - Implementation of **probabilistic record linkage**.  
  - Use of **entity resolution models** to determine if two values should belong to the same attribute.  
- **Status**: Planned.

---

### Stage 4: AI Framework Implementation and Scalability/Maintenance Strategy

- **Objective**:  
  Develop a unified **AI framework** to address multiple data quality issues concurrently.  
- **Features**:  
  - Integration of **model fusion** to combine solutions from earlier stages.  
  - Explainable AI (XAI) for transparency and enhanced decision-making.  
  - Development of a **client software tool** for streamlined usage.  
- **Status**: Planned.

---

### Stage 5: Real-time Data Ingestion and Quality Testing

- **Objective**:  
  End-to-end validation of the AI framework from a **data engineer's perspective**.  
- **Features**:  
  - Efficient approaches for **real-time data ingestion** and quality testing.  
  - Application to fresh datasets for comprehensive validation.  
- **Status**: Planned.

---

### Stage 6: Embedding of the Knowledge Exchange

- **Objective**:  
  Build capacity for the client organization to take full ownership of the developed tools.  
- **Features**:  
  - Development of a **Knowledge Base (KB)** with detailed documentation and guides.  
  - Post-project training and support for tool adaptation and scalability.  
- **Status**: Planned.

---

## Installation

### Clone the Repository
```
git clone https://github.com/itsbigspark/ntu-ktp-data-quality.git
cd ntu-ktp-data-quality
```

```
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Data Validator

- **Command Line Interface:**
  
```
cd Data_Validator
python main.py clean_data.csv unclean_data.csv --rules custom_rules.json
```

- **Streamlit App:**
  
```
streamlit run appy.py
```

### 2. Data Cleaner

- **Command Line Interface:**

```
cd Data_Cleaner
python main.py error_data.csv reference_data.csv
```

- **Streamlit App:**

```
streamlit run data_cleaner_app.py
```

### 3. Duplicate Detection Tool

- **Command Line Interface:**

```
cd DuplicateDetectionTool
python main.py input_dataset.csv
```
- **Streamlit App:**
bash

```
streamlit run duplicates_app.py
```

---

## Contributors
- **Preetham Yuvaraj**
- **BigSpark Team**

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or support:

Email: preetham.yuvaraju@bigspark.dev


