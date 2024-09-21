import ollama
import csv
import json
import re

print("Script started...")

# Specify the paths and model
input_csv_path = '/Users/lachyshinnick/Downloads/valid_reports.csv'  # Replace with your input CSV file path
output_csv_path = '/Users/lachyshinnick/Desktop/codes/ollamaTest/final_results.csv'  # Replace with your output CSV file path
error_csv_path = '/Users/lachyshinnick/Desktop/codes/ollamaTest/error.csv'  # Replace with your error file path
desiredModel = 'llama3.1:latest'

# Function to extract JSON from model response
def extract_json_from_response(response_text):
    try:
        # Ensure response is cleanly formatted and extract the first JSON object found
        json_matches = re.findall(r'({.*?})', response_text, re.DOTALL)
        if json_matches:
            json_str = json_matches[-1].replace("'", '"')  # Fix apostrophes for valid JSON
            return json.loads(json_str)
        else:
            return {"Error": "Invalid JSON"}
    except (ValueError, json.JSONDecodeError):
        return {"Error": "Invalid JSON"}

# Function to clean findings output
def clean_findings(findings):
    if isinstance(findings, list):
        # Join the list into a single string
        cleaned_findings = "; ".join(findings)
    else:
        # In case it's a string, directly use it
        cleaned_findings = findings
    
    # Remove unwanted symbols, if any (e.g., square brackets or apostrophes)
    cleaned_findings = re.sub(r"[\[\]']", "", cleaned_findings)
    
    # Split the findings into individual statements
    statements = cleaned_findings.split('; ')
    updated_statements = []
    for statement in statements:
        # Replace "There are" with "There is" at the beginning of the statement
        updated_statement = re.sub(r'^There are', 'There is', statement)
        updated_statements.append(updated_statement)
    
    # Rejoin the statements
    final_findings = '; '.join(updated_statements)
    
    return final_findings

def extract_findings(report_content, file_name):
    prompt = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful assistant trained to extract key medical findings from radiology reports.

        **Instructions**:

        - **Output Format**: Provide findings in JSON format with the key "Findings". The value should be a string containing one or more statements.
        - **Statement Format**:
        - For **positive findings**:
            - If the location is mentioned: "There is <diagnosis> located at <location>."
            - If the location is not mentioned: "There is <diagnosis>."
        - For **negative findings** (only if explicitly mentioned in the report):
            - "There is no <diagnosis>."
        - For **normal findings**:
            - If explicitly mentioned in the report, include them as: "There is normal <structure>."
        - **General Rules**:
        - **Include** all findings mentioned in the report, both abnormal and normal.
        - **Do not** include speculative or uncertain findings. Exclude any statements containing words like "possible", "suggests", "may indicate", "could", "probably", etc.
        - **Do not** include redundant or repetitive statements.
        - **Ensure** that each statement starts with "There is", "There is no", or "There is normal" as per the format.
        - **Only include** findings that strictly adhere to the required format.

        **Examples**:

        **Example 1**:

        Report Content:
        "there is a fracture involving the proximal shaft of the right humerus. there is minimal displacement. the humeral head is enlocated. alignment at the elbow joint is anatomical."

        Findings:
        {{
        "Findings": "There is a fracture located at the proximal shaft of the right humerus; There is minimal displacement; There is normal humeral head; There is normal alignment at the elbow joint."
        }}

        **Example 1 note**: This output effectively captures all key findings about the fracture, displacement, and normal alignment. It follows the "There is" format for each positive finding and maintains conciseness while including the most relevant diagnostic information.

        **Example 2**:

        Report Content:
        "the cardiomediastinal contour appears normal. the lungs and pleural spaces are clear."

        Findings:
        {{
        "Findings": "There is normal cardiomediastinal contour; There is clear lungs; There is clear pleural spaces."
        }}

        **Example 2 note**: This output is short and precise, capturing the normal findings of the cardiomediastinal contour, lungs, and pleural spaces. The "There is" format is used effectively to ensure clarity and consistency.

        **Example 3**:

        Report Content:
        "no orbital floor fracture. no maxillary sinus fluid level. no nasal bone fracture. no evidence of zygomatic arch fracture."

        Findings:
        {{
        "Findings": "There is no orbital floor fracture; There is no maxillary sinus fluid level; There is no nasal bone fracture; There is no zygomatic arch fracture."
        }}

        **Example 3 note**: This output concisely captures multiple negative findings. The use of "There is no" clearly communicates the absence of fractures and fluid levels, maintaining adherence to the required format without unnecessary details.

        **Example 4**:

        Report Content:
        "the lungs are clear. heart size is normal. no pleural effusion."

        Findings:
        {{
        "Findings": "There is clear lungs; There is normal heart size; There is no pleural effusion."
        }}

        **Example 4 note**: This output succinctly identifies key positive and negative findings about the lungs, heart size, and pleural effusion. The response follows the required format and avoids unnecessary elaboration, keeping the information relevant and concise.

        **Example 5**:

        Report Content:
        "there is a possible fracture of the distal radius. appearances suggest a mild sprain. no definite fracture is seen."

        Findings:
        {{
        "Findings": "" 
        }}

        **Example 5 note**: Speculative language such as "possible" and "suggest" is excluded, as it does not meet the requirement for definitive diagnostic statements. This response correctly omits findings that are uncertain or speculative.

        **Explanation**: Speculative findings containing words like "possible" and "suggest" are omitted, and relevant medical information is included.

        <|eot_id|><|start_header_id|>user<|end_header_id|>

        **Report Content**:
        "{report_content}"

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = ollama.chat(model=desiredModel, messages=[
                {'role': 'user', 'content': prompt},
            ])
            raw_content = response['message']['content'].strip()
            # Extract JSON
            print(raw_content)
            structured_data = extract_json_from_response(raw_content)
            if "Error" not in structured_data:
                # Successfully extracted findings
                return structured_data
            else:
                print(f"Attempt {attempt+1} failed due to invalid JSON in findings for {file_name}")
        except Exception as e:
            print(f"Attempt {attempt+1} failed processing findings for file {file_name}: {e}")
    # After retries, return error
    print(f"Skipping sentence after {max_retries} attempts in file {file_name}")
    structured_data = {"Error": "Failed after retries"}
    return structured_data

# Function to process the CSV file
def process_csv_file(input_csv_path, output_csv_path):
    # Read the input CSV file
    with open(input_csv_path, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['Key Findings']
        
        # Open the output CSV file for writing (overwrite mode)
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as outfile, \
             open(error_csv_path, 'w', encoding='utf-8', newline='') as errorfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            error_writer = csv.DictWriter(errorfile, fieldnames=reader.fieldnames)
            error_writer.writeheader()
            
            for row in reader:
                file_name = row.get('body_part_file_name', 'Unknown')
                report_content = row.get('report_content', '')
                
                if not report_content:
                    print(f"No report content for file {file_name}. Skipping.")
                    continue
                
                # Extract findings from the entire report content
                findings_output = extract_findings(report_content, file_name)
                
                if "Error" in findings_output:
                    print(f"Writing file {file_name} to error.csv due to error.")
                    error_writer.writerow(row)
                    errorfile.flush()
                    continue
                
                # Get findings
                findings = findings_output.get("Findings", "")
                
                # Clean the findings to remove unwanted symbols
                cleaned_findings = clean_findings(findings)
                
                # Add the cleaned findings to the row
                row['Key Findings'] = cleaned_findings
                
                # Write the row to the output CSV
                writer.writerow(row)
                outfile.flush()
                
        print(f"All findings saved to {output_csv_path}")
        print(f"Errors saved to {error_csv_path}")

# Call the function to process the CSV file
process_csv_file(input_csv_path, output_csv_path)