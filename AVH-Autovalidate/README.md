# Auto-Validate-by-History and Pattern Matching for Validation of UK Companies House Data


## Overview
This Jupyter notebook project analyzes and validates data from the UK Companies House. It performs data loading from AWS S3, exploratory data analysis (EDA), data cleaning, and validation using derived historic metrics regex patterns. The project is part of the ntu-ktp-data-quality initiative.

## Dataset
The project uses the Companies House dataset, which contains information about registered companies in the UK.

- **Dataset Source**: [Companies House - Free Company Data Product](http://download.companieshouse.gov.uk/en_output.html)
- **File Format**: CSV and Parquet
- **File Names**:
  - `Company_sample2.csv`
  - `companies.parquet`

## Repository Structure
- `AVH_AutoValidate_CompaniesHouse.ipynb`: Main Jupyter notebook containing the analysis
- `requirements.txt`: List of Python packages required for this project
- `README.md`: This file, containing project information and instructions

## Prerequisites
- Python 3.x
- Jupyter Notebook or JupyterLab
- AWS account with S3 access
- AWS CLI
- Git (for cloning the repository)

## Setup and Installation

1. Clone the ntu-ktp-data-quality repository:
   git clone https://github.com/your-username/ntu-ktp-data-quality.git
2. Navigate to the project directory:
   cd ntu-ktp-data-quality/AVH-Autovalidate
3. Create and activate a virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
4. Install the required packages:
   pip install -r requirements.txt
5. Set up AWS credentials:
- Run `aws configure` and enter your AWS credentials when prompted

## Data Preparation

1. Download the dataset:
- Visit the [Companies House data download page](http://download.companieshouse.gov.uk/en_output.html)
- Download the `BasicCompanyDataAsOneFile-*.zip` file

2. Extract the ZIP file and locate the CSV file (rename it to `Company_sample2.csv` if needed)

3. Convert the CSV to Parquet format (optional, if you need the Parquet file):
   ```python
   import pandas as pd

   df = pd.read_csv('Company_sample2.csv')
   df.to_parquet('companies.parquet')
4. Upload the files to your S3 bucket:
   aws s3 cp Company_sample2.csv s3://your-bucket-name/Companies_data/
   aws s3 cp companies.parquet s3://your-bucket-name/Companies_data/

## Usage

Launch Jupyter Notebook:
Copyjupyter notebook

In the Jupyter interface, navigate to and open Companies_House_Data_Analysis.ipynb.
Before running the notebook, update the following in the first few cells:

S3 bucket name
Input file paths
Output file paths


Run the cells in order, following any instructions within the notebook.

Notebook Workflow

Data Loading:

The notebook uses boto3 to connect to S3 and read the data files
CSV file is read using pandas
Parquet file is read using pyspark


Exploratory Data Analysis (EDA):

Examines dataset structure
Generates summary statistics
Analyzes missing values
Creates visualizations


Data Validation:

Applies regex patterns to validate data format
Patterns are defined for each column based on expected data format


Data Cleaning:

Processes and cleans the data based on validation results
Handles missing values and format inconsistencies


Reporting:

Generates reports on data issues and cleaned data
Creates Excel files with highlighted issues


Output:

Saves reports back to S3 using boto3



Output
The notebook generates two main reports:

highlighted_issues_report.xlsx: Highlights issues in the original dataset
cleaned_report.xlsx: Shows the cleaned dataset with any remaining issues highlighted

Both reports are saved in the specified S3 bucket under the 'Companies_Clean_report' folder.
Troubleshooting

If you encounter issues with AWS access, ensure your AWS CLI is correctly configured with valid credentials.
For package-related issues, make sure all dependencies are correctly installed via the requirements.txt file.
If you have trouble reading the Parquet file, ensure you have the necessary dependencies installed (pyarrow for pandas, or pyspark if using PySpark).

## Contributing

This project is part of the ntu-ktp-data-quality initiative. If you're a member of the project team or a student working on this assignment:

1. Ensure you have access to the repository.
2. Clone the repository to your local machine.
3. Create a new branch for your work:
   git checkout -b your-name/feature-description
4. Make your changes and commit them with descriptive commit messages.
5. Push your changes to the repository:
   git push origin your-name/feature-description
6. If required, create a pull request for your changes to be reviewed and merged.

For external contributors or if you've found an issue:
- Please contact the project maintainer at [maintainer's email] to discuss potential contributions or report issues.

Note: Always follow the project's coding standards and guidelines when making changes.

## License 
This project is licensed under the MIT License - see the LICENSE file for details.
Contact
Your Name - preetham.yuvaraju@bigspark.dev
Project Link: https://github.com/itsbigspark/ntu-ktp-data-quality
