# AI-Based Fraud Management System for UID Aadhaar

## Overview
This project implements an **AI-Based Fraud Management System** aimed at detecting and classifying fraudulent activities associated with UID Aadhaar data. The system leverages advanced machine learning techniques to identify anomalies and potential fraud, ensuring enhanced security and reliability for Aadhaar services.

## Key Features
- **Fraud Classification Model**:
  - Identifies whether given data indicates a potential fraud or legitimate use.
  - Provides insights into the type and nature of detected fraud.
- **Fraud Detection Model**:
  - Detects anomalies in Aadhaar transactions and flags suspicious activities.
  - Employs state-of-the-art techniques for high accuracy and precision.
- **Scalability**:
  - Designed to handle large datasets efficiently.
- **User-Friendly Interface**:
  - Interactive and intuitive interface for reviewing results and insights.

## Project Structure
The project consists of two main Jupyter Notebooks:

1. **Fraud Classification Notebook**
   - Implements a classification model to categorize data points as fraudulent or non-fraudulent.
   - Includes data preprocessing, feature engineering, model training, and evaluation.

2. **Fraud Detection Notebook**
   - Focuses on anomaly detection using machine learning algorithms.
   - Covers preprocessing, unsupervised learning techniques, and evaluation.

## Setup Instructions
1. Clone this repository or download the project files.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset and place it in the appropriate folder (`data/` by default).
4. Open the Jupyter Notebooks in your preferred environment and execute the cells sequentially.

## Prerequisites
- Python 3.8+
- Jupyter Notebook
- Libraries:
  - NumPy
  - Pandas
  - Scikit-learn

## Working
1. **Data Preprocessing**:
   - Handles missing values, data normalization, and feature extraction.
2. **Model Training**:
   - For the classification notebook, supervised learning algorithms like Random Forest, Logistic Regression, or XGBoost are used.
   - For the detection notebook, unsupervised algorithms like Isolation Forest or DBSCAN are implemented.
3. **Evaluation**:
   - Models are evaluated using metrics such as accuracy, precision, recall, and F1-score for classification.
   - For anomaly detection, techniques like ROC-AUC and confusion matrices are employed.
## How It Works
1. **Document Upload**:  
   - Upload input Excel files and a folder containing document images.
   
2. **Document Classification**:  
   - Uses YOLO to classify documents as **“Aadhar”** or **“Non-Aadhar.”**
   
3. **Text Extraction**:  
   - Detects and extracts critical fields from **“Aadhar”** documents.
   
4. **Data Matching**:  
   - Matches extracted fields with input Excel records and calculates similarity scores.
   
5. **Result Export**:  
   - Outputs results into an Excel file, including:  
     - Document classification.  
     - Extracted Name, Address, UID.  
     - Matching scores for Name, Address, and UID.  

## Outputs

### Processed Excel File:
The exported Excel file contains the following columns:
- **SrNo**  
- **Document Type** (e.g., "Aadhar", "Non-Aadhar")  
- **Extracted Name**  
- **Extracted Address**  
- **Extracted UID**  
- **Name Match Score**  
- **Overall Match Score**  

### Streamlit Dashboard:
- An interactive dashboard displays processing results in real-time.  

## Example Workflow

1. **Input**:  
   - An Excel file containing records to be matched.  
   - A folder containing document images.  

2. **Processing**:  
   - Classifies documents into categories such as "Aadhar" or "Non-Aadhar."  
   - Extracts text from critical fields like Name, Address, and UID.  
   - Matches extracted data with the records provided in the Excel file.  

3. **Output**:  
   - An exported Excel file containing:  
     - Document classification results.  
     - Extracted Name, Address, UID.  
     - Matching scores for each record.  


## Usage
- Load your dataset into the provided format and execute the notebooks.
- Review model outputs and insights to identify fraudulent activities.
- Modify parameters or algorithms as needed to improve performance.

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`feature/my-feature`).
3. Commit your changes and push to your branch.
4. Submit a pull request with a detailed description.

## Acknowledgments
- Developers and contributors who have built the core framework.
- Open-source libraries and tools that made this implementation possible.

