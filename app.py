import os
import re
import pandas as pd
import cv2
import easyocr
from difflib import SequenceMatcher
from ultralytics import YOLO
import streamlit as st

# ---- Utility Functions ----
ignore_terms = [
    "PO-", "PO", "Marg", "Peeth", "Veedhi", "Rd", "Lane", "NR",
    "Beside", "Opposite", "OPP", "Behind", "near", "Enclave",
    "Township", "Society", "Soc", "Towers", "Block", "S/o", "C/o",
    "D/o", "W/o",
]

def remove_ignore_terms(address):
    """Remove stopwords and non-alphanumeric characters from the address."""
    if not isinstance(address, str):
        return ""
    for term in ignore_terms:
        address = re.sub(r'\b' + re.escape(term) + r'\b', '', address, flags=re.IGNORECASE)
    address = re.sub(r'[^a-zA-Z0-9\s]', '', address)
    address = re.sub(r'\s+', ' ', address).strip()
    return address

def calculate_similarity(str1, str2):
    """Calculate similarity score between two strings."""
    return SequenceMatcher(None, str1, str2).ratio()

def normalize_address(address):
    """Normalize address for comparison."""
    return remove_ignore_terms(address).lower()

def exact_letter_match(name1, name2):
    """Check for exact letter match."""
    return name1.lower() == name2.lower()

def name_match(input_name, extracted_name):
    """Match names using various rules."""
    return calculate_similarity(input_name.lower(), extracted_name.lower()) * 100

def process_folder(folder_path):
    """Process images in a folder to classify documents and extract text."""
    results = {}
    model_classify = YOLO('/Users/akashngowda/runs/classify/train/weights/best.pt')
    model_detect = YOLO("/System/Volumes/Data/Users/akashngowda/runs/detect/train22/weights/best.pt")
    reader = easyocr.Reader(['en'])

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            base_serial_number = re.sub(r'(_\d+)?\.\w+$', '', filename)
            results[base_serial_number] = {"status": "failure", "data": {}, "classification": "Non-Aadhar"}

            classify_results = model_classify.predict(image_path)
            predicted_class = None

            try:
                if hasattr(classify_results[0].probs.top1, 'name'):
                    predicted_class = classify_results[0].probs.top1.name
                else:
                    predicted_class_index = classify_results[0].probs.top1
                    predicted_class = model_classify.names[predicted_class_index]
            except AttributeError:
                st.warning(f"Error processing file: {filename}. Prediction format unexpected.")

            if predicted_class and predicted_class.lower() == "aadhar":
                results[base_serial_number]["classification"] = "Aadhar"
                detect_results = model_detect(image_path)
                image = cv2.imread(image_path)
                extracted_data = {}

                for result in detect_results[0].boxes.data.tolist():
                    x1, y1, x2, y2, _, class_id = map(int, result[:6])
                    field_class = model_detect.names[class_id]
                    cropped_roi = image[y1:y2, x1:x2]
                    gray_roi = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)
                    text = reader.readtext(gray_roi, detail=0)
                    extracted_data[field_class] = ' '.join(text)

                results[base_serial_number]["status"] = "success"
                results[base_serial_number]["data"] = extracted_data

    return results

def process_and_match_addresses(input_file, output_file, folder_path):
    """Process the input Excel file and folder for matching."""
    extracted_data = process_folder(folder_path)
    df = pd.read_excel(input_file)

    # Add result columns
    df["Document Type"] = "Unknown"
    df["Extracted Address"] = ""
    df["Extracted Name"] = ""
    df["Extracted UID"] = ""
    df["Overall Match Score"] = 0
    df["Name Match Score"] = 0

    # Process rows
    for idx, row in df.iterrows():
        sr_no = str(row["SrNo"])
        extracted_info = extracted_data.get(sr_no, {})
        classification = extracted_info.get("classification", "Non-Aadhar")
        df.at[idx, "Document Type"] = classification

        # Update extracted data
        extracted_address = extracted_info.get("data", {}).get("address", "")
        extracted_name = extracted_info.get("data", {}).get("name", "")
        extracted_uid = extracted_info.get("data", {}).get("uid", "")
        df.at[idx, "Extracted Address"] = extracted_address
        df.at[idx, "Extracted Name"] = extracted_name
        df.at[idx, "Extracted UID"] = extracted_uid

        if classification == "Aadhar":
            name_score = name_match(row.get("Name", ""), extracted_name)
            df.at[idx, "Name Match Score"] = name_score

            # Calculate overall score (if needed for further processing)
            df.at[idx, "Overall Match Score"] = name_score  # Example for demonstration

    # Save results
    df.to_excel(output_file, index=False, engine="openpyxl")
    st.success(f"Processing complete. Results saved to {output_file}")

# ---- Streamlit UI ----
st.title("Document Classification and Matching Tool")

uploaded_file = st.file_uploader("Upload Input Excel File", type="xlsx")
uploaded_folder = st.text_input("Upload Document Folder Path")
output_file_path = st.text_input("Output File Path", "output.xlsx")

if st.button("Process"):
    if uploaded_file and uploaded_folder:
        with st.spinner("Processing..."):
            process_and_match_addresses(uploaded_file, output_file_path, uploaded_folder)
        st.success(f"Processing complete. Results saved to {output_file_path}")
    else:
        st.error("Please upload all required inputs.")
