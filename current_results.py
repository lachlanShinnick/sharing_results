import ollama
import csv
import json
import re
from difflib import SequenceMatcher

print("Script started...")

# Specify the paths and models
input_csv_path = '/Users/lachyshinnick/Downloads/valid_reports.csv'  # Replace with your input CSV file path
output_csv_path = '/Users/lachyshinnick/Desktop/codes/ollamaTest/final_results.csv'  # Replace with your output CSV file path
error_csv_path = '/Users/lachyshinnick/Desktop/codes/ollamaTest/error.csv'  # Replace with your error file path

# List of temperature-specific models
temp_models = [
    'temp_0.0:latest',
    'temp_0.2:latest',
    'temp_0.4:latest',
    'temp_0.6:latest',
    'temp_0.8:latest',
    'temp_1.0:latest'
]

# Base model for selection
selection_model = 'llama3.1:latest'

def is_similar(statement1, statement2, threshold=0.99):
    return SequenceMatcher(None, statement1, statement2).ratio() > threshold

# Function to extract JSON from model response
def extract_json_from_response(response_text):
    try:
        # Extract the first JSON object found
        json_matches = re.findall(r'({.*?})', response_text, re.DOTALL)
        if json_matches:
            json_str = json_matches[-1].replace("'", '"')  # Replace single quotes with double quotes
            return json.loads(json_str)
        else:
            return {"Error": "Invalid JSON"}
    except (ValueError, json.JSONDecodeError) as e:
        print(f"JSON decoding error: {e}")
        return {"Error": "Invalid JSON"}

# Function to clean and consolidate findings
def clean_findings(findings):
    if isinstance(findings, list):
        cleaned_findings = "; ".join(findings)
    else:
        cleaned_findings = findings

    # Remove unwanted symbols
    cleaned_findings = re.sub(r"[\[\]']", "", cleaned_findings)
    # Remove random full stops (but not the ones at the end of sentences)
    cleaned_findings = re.sub(r'\.(?!\s|$)', '', cleaned_findings)

    # Split the findings into individual statements
    statements = re.split(r';\s*', cleaned_findings)
    updated_statements = []
    seen_statements = set()
    for statement in statements:
        # Replace "There are" with "There is" at the beginning of the statement
        updated_statement = re.sub(r'^There are', 'There is', statement.strip())
        # Avoid duplicates
        if updated_statement.lower() not in seen_statements:
            seen_statements.add(updated_statement.lower())
            updated_statements.append(updated_statement)

    # Rejoin the statements
    final_findings = '; '.join(updated_statements)
    return final_findings

# Function to prompt all temperature models with a sentence
def prompt_temp_models(report_content, sentence):
    outputs = []
    for model_name in temp_models:
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
            - If the location is mentioned: "There is no <diagnosis> located at <location>."
            - If the location is not mentioned: "There is <diagnosis>."
            - For **normal findings**:
            - If explicitly mentioned in the report, include them as: "There is normal <structure>."
            - If the term "unremarkable" is used to describe a body part, interpret this as "There is no abnormality at <body part>."
        - **General Rules**:
        - **Do not hallucinate**: Only include information explicitly mentioned in the report. Do not make any assumptions or create findings that are not directly stated.
        - **One finding per sentence**: Do not combine findings into a single sentence using conjunctions like "and" or "with". Each statement should describe one finding at one location.
        - **If a sentence in the report says "There is <condition-1> and <condition-2> at <location>", split it into two separate findings like: "There is <condition-1> at <location>; There is <condition-2> at <location>".**
        - **Do not use conjunctions like "and" or "with" to connect multiple findings**. Instead, separate each finding into its own sentence.
        - **Use nouns instead of adverbs** to describe locations. For example, replace "subdiaphragmatically" with "at sub-diaphragm".
        - **Remove adjectives describing severity or visibility** unless they are clinically significant. For example, remove words like "mild" or "conspicuous" unless explicitly required for diagnosis.
        - **Include** all findings mentioned in the report, both abnormal and normal.
        - **Do not** include speculative or uncertain findings. Exclude any statements containing words like "possible", "suggests", "may indicate", "could", "probably", etc.
        - **Do not** include redundant or repetitive statements.
        - **Do not** include any references to previous X-rays or imaging results. Focus solely on the current medical findings and diagnosis in the report.
        - **Ensure** that each statement starts with "There is", "There is no", or "There is normal" as per the format.
        - **Only include** findings that strictly adhere to the required format and information given in the report.
        - **Use exact medical terminology from the report without substitutions or synonyms**
        - **Do not include possible findings, and do not include severity of the degree of findings. Remove words such as "mild", "acute", "minimal", and other descriptors that indicate the extent of the condition.


**Examples**:

**Example 1**:

Sentence:
"There is a greenstick fracture of the distal radius."

Findings:
{{
"Findings": "There is a greenstick fracture located at distal radius;"
}}

**Example 1 note**: The finding is expressed concisely, beginning with “There is” to follow the standard format. The phrase “located at distal radius” clarifies the specific location of the fracture, providing a clear and structured description.

**Example 2**:

Sentence:
"there is a conspicuous gas-filled and dilated bowel loop located at the left mid-abdomen."

Findings:
{{
    "Findings": "There is a gas-filled bowel loop located at the left mid-abdomen; There is a dilated bowel loop located at the left mid-abdomen."
}}

**Example 2 note**: The adjective "conspicuous" is removed as it is unnecessary. The phrase "gas-filled and dilated bowel loop" is split into two sentences to ensure only one finding per sentence. The word "and" is avoided, and the sentence structure is simplified. The word "mildly" is removed as we are not concerned with the degree of the diagnosis.

**Example 3**:

Sentence:
"the lungs are clear, heart size is normal and no pleural effusion."

Findings:
{{
"Findings": "There is clear lungs; There is normal heart size; There is no pleural effusion;"
}}

**Example 3 note**: The sentence is split into three distinct findings, each starting with “There is” or “There is no,” ensuring clarity and adherence to the preferred format. This avoids using “and” and provides a clear, standardized description of each observation.

**Example 4**:

Sentence:
"There is minimal displacement of the fracture at the right humerus."

Findings:
{{
"Findings": "There is displacement of fracture located at right humerus;"
}}

**Example 4 note**: The descriptor “minimal” is removed as we are not concerned with the degree of the diagnosis, only its presence. The finding is expressed as “There is displacement of fracture,” and the location “right humerus” is included to provide clarity.

**Example 5**:

Sentence:
"The humeral head is enlocated."

Findings:
{{
"Findings": "There is an enlocated humeral head;"
}}

**Example 5 note**: The term “enlocated” indicates normal positioning and should be expressed as “There is an enlocated humeral head.."

**Example 6**:

Report Content:
"there is central bronchial wall thickening and a mild peribronchial interstitial opacity located at upper lower zones."

Findings:
{{
"Findings": "There is central bronchial wall thickening located at upper lower zones; There is a peribronchial interstitial opacity located at upper lower zones."
}}

**Example 3 note**: The adjective "mild" is removed as it does not contribute to the essential finding. The findings are separated into two sentences, with each describing one distinct condition. The conjunction "and" is removed to prevent the combination of findings into one sentence.

**Example 7**:

Sentence:
"This could reflect localized ileus."

Findings:
{{
"Findings": ""
}}

**Example 7 note**: This does not include any findings, because the findings are not definite."

**Report Content** (only for context):
{report_content}

---

Focus solely on summarising this sentence as per the instructions above. Ensure each findings starts with "there is", even if it does not make perfect grammatical sense to do so.
"{sentence}"

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

        try:
            response = ollama.chat(model=model_name, messages=[
                {'role': 'user', 'content': prompt}
            ])
            raw_content = response['message']['content'].strip()
            print(f"Model {model_name} response: {raw_content}")
            # Extract JSON
            structured_data = extract_json_from_response(raw_content)
            if "Error" not in structured_data:
                findings = structured_data.get("Findings", "")
                if findings:
                    # Collect the full JSON output
                    outputs.append(structured_data)
            else:
                print(f"Error: Invalid JSON from model {model_name} for sentence: {sentence}")
        except Exception as e:
            print(f"Exception while prompting model {model_name}: {e}")

    return outputs

# Function to let the LM select the best output for a single sentence
def select_best_output(outputs, selection_model, sentence):
    if not outputs:
        return ""

    # Create a prompt for the LM to select the most accurate and complete medical findings
    selection_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>.\n\n"

    # Enhanced instructions for selecting the best option, including the original sentence
    selection_prompt += f"""
You are an expert assistant trained to select the most accurate and properly formatted medical findings from the options provided below.

**Instructions**:

- **Original Sentence**:
  "{sentence}"

- **Criteria for Selection**:
  1. **Adherence to Statement Formats**:
      - **Positive Findings**:
          - Must start with "There is <diagnosis> located at <location>." if location is mentioned.
          - Must start with "There is <diagnosis>." if location is not mentioned.
      - **Negative Findings**:
          - Must start with "There is no <diagnosis> located at <location>." if location is mentioned.
          - Must start with "There is no <diagnosis>." if location is not mentioned.
  2. **Information Completeness and Relevance**:
      - Select the finding that includes all relevant findings from the original sentence without omissions.
      - Select the finding such that no additional or irrelevant findings should be added.
      - The information you select should add some knowledge.
  3. **Clarity and Precision**:
      - The selected statement should be clear, precise, and free from ambiguity.
      - The selected statement should be split up as much as possible, so that each sentence has its own point.
  4. **No Speculative or Redundant Information**:
      - Exclude any findings that are speculative or uncertain. Specifically, do not include findings containing words such as ‘probably,’ ‘probable,’ ‘suggests,’ ‘possible,’ ‘may indicate,’ or any similar terms that imply uncertainty.
      - Avoid redundant or repetitive statements.
      - If none of the statements align with the criteria, output the number 9
  5. **Alignment with Original Language**:
      - Select the statement that best adheres to the original language in the original statement only if it aligns with the above statements.
      

- **Response Requirement**:
  Respond with only the number of the selected option (e.g., '1', '2', etc.) and nothing else.

**examples**:

**Example 1**:

**Original Sentence**:
"There is a greenstick fracture of the distal radius."

**Options**:
Option 1: {{"Findings": "There is a greenstick fracture located at distal radius;"}}
Option 2: {{"Findings": "There is a fracture at distal radius."}}

**Best Option**:
1

**Explanation**:
Option 1 provides a more detailed and precise finding by specifying the type of fracture ("greenstick") and its exact location ("distal radius"). This aligns with the criteria of clarity and completeness.

---

**Example 2**:

**Original Sentence**:
"there is a conspicuous gas-filled and dilated bowel loop located at the left mid-abdomen."

**Options**:
Option 1: {{"Findings": "There is a gas-filled bowel loop located at the left mid-abdomen; There is a dilated bowel loop located at the left mid-abdomen."}}
Option 2: {{"Findings": "There is a conspicuous gas-filled and dilated bowel loop located at the left mid-abdomen."}}

**Best Option**:
1

**Explanation**:
Option 1 splits the findings into two separate statements, enhancing clarity by addressing each condition individually. It also removes the unnecessary adjective "conspicuous," adhering to the general rules.

---

**Example 3**:

**Original Sentence**:
"no acute bony or soft tissue anomaly."

**Options**:
Option 1: {{"Findings": "There is no acute bony or soft tissue anomaly."}}
Option 2: {{"Findings": "There is no acute bony anomaly; There is no acute soft tissue anomaly;"}}

**Best Option**:
2

**Explanation**:
Option 2 splits the negative findings into two separate statements, ensuring each finding is clear and follows the one finding per statement rule.

---

**Example 4**:

**Original Sentence**:
"a small gas locule persists here."

**Options**:
Option 1: {{"Findings": "There is a gas locule located at the right lateral chest wall;"}}
Option 2: {{"Findings": "There is a small gas locule located at unspecified location;"}}

**Best Option**:
1

**Explanation**:
Option 1 provides a specific location for the gas locule, making the finding more informative and actionable compared to the vague "unspecified location."

---

**Example 5**:

**Original Sentence**:
"there is associated mild right lower lobe collapse."

**Options**:
Option 1: {{"Findings": "There is a collapse located at right lower lobe;"}}
Option 2: {{"Findings": "There is right lower lobe collapse; There is mild right lower lobe collapse;"}}

**Best Option**:
1

**Explanation**:
Option 1 accurately captures the finding without redundancy. Option 2 repeats the same condition by stating both “right lower lobe collapse” and “mild right lower lobe collapse,” which does not add any additional meaningful information. Removing the adjective “mild” ensures that the finding remains clear and concise.

---

**New Task**:

**Original Sentence**:
"{sentence}"

**Options**:
"""

    # Append the options with proper JSON formatting
    for idx, output_dict in enumerate(outputs, start=1):
        # Ensure that each option is a JSON-formatted string
        escaped_output = json.dumps(output_dict)
        selection_prompt += f"Option {idx}: {escaped_output}\n"

    # Append the response requirement again to reinforce it after options
    selection_prompt += """
**Response Requirement**:
Respond with the number of the selected option (e.g., '1', '2', etc.) followed by a detailed explanation of why you chose this option, referencing both the original sentence and the choice.
Ensure you are picking finings that follow the criteria, and choose the sentences that split up the findings into singular sentences.
Prefer outputs that have "located at" over outputs that have "at".
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    try:
        # Send the selection prompt to the language model
        response = ollama.chat(model=selection_model, messages=[
            {'role': 'user', 'content': selection_prompt}
        ])
        best_output = response['message']['content'].strip()
        print(f"Selected best output (raw response): {best_output}")

        # Extract the number from the response using regex for options 1 to 6
        match = re.search(r'\b([1-6])\b', best_output)
        if match:
            selected_number = int(match.group(1))
            if 1 <= selected_number <= len(outputs):
                selected_finding_dict = outputs[selected_number - 1]
                selected_finding = selected_finding_dict.get("Findings", "")
                print(f"Selected option {selected_number}: {selected_finding}")
                return selected_finding
            else:
                print(f"Selected number {selected_number} is out of range.")
                return ""
        else:
            print("No valid option number found in the response.")
            return ""
    except Exception as e:
        print(f"Exception while selecting the best output: {e}")
        return ""

# Main function to extract findings for an entire report
def extract_findings(report_content, file_name):
    # Split the report into sentences based on periods
    sentences = [s.strip() for s in re.split(r'\.\s*', report_content.strip()) if s.strip()]
    all_findings = []

    for sentence in sentences:
        print(f"\nProcessing sentence: {sentence}")
        # Get outputs from all temperature models
        temp_outputs = prompt_temp_models(report_content, sentence)
        if not temp_outputs:
            print(f"No outputs from temperature models for sentence: {sentence}")
            continue

        # Use LM to select the best output, providing the original sentence
        best_output = select_best_output(temp_outputs, selection_model, sentence)
        if best_output:
            # Clean the selected output
            cleaned_output = clean_findings(best_output)
            if cleaned_output:
                all_findings.append(cleaned_output)
                print(f"Added findings: {cleaned_output}")
            else:
                print(f"No valid findings in the best output for sentence: {sentence}")
        else:
            print(f"No best output selected for sentence: {sentence}")

    if all_findings:
        # Remove duplicates while preserving order
        seen = set()
        unique_findings = []
        for finding in all_findings:
            if finding.lower() not in seen:
                seen.add(finding.lower())
                unique_findings.append(finding)
        # Return the consolidated findings
        consolidated_findings = "; ".join(unique_findings)
        print(f"\nConsolidated Findings: {consolidated_findings}")
        return {"Findings": consolidated_findings}
    else:
        # Return empty findings if none are extracted
        print("\nNo findings extracted from the report.")
        return {"Findings": ""}

# Function to process the CSV file
def process_csv_file(input_csv_path, output_csv_path):
    # Open the input CSV file
    with open(input_csv_path, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        # Add 'Findings' to the fieldnames if not already present
        fieldnames = reader.fieldnames + ['Findings'] if 'Findings' not in reader.fieldnames else reader.fieldnames

        # Open the output and error CSV files for writing
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as outfile, \
             open(error_csv_path, 'w', encoding='utf-8', newline='') as errorfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            error_writer = csv.DictWriter(errorfile, fieldnames=reader.fieldnames)
            error_writer.writeheader()

            # Process each row in the input CSV
            for row in reader:
                file_name = row.get('body_part_file_name', 'Unknown')
                report_content = row.get('report_content', '')

                if not report_content:
                    print(f"No report content for file {file_name}. Skipping.")
                    continue

                # Extract findings from the entire report content
                findings_output = extract_findings(report_content, file_name)

                if "Error" in findings_output:
                    print(f"Writing file {file_name} to error.csv due to error in extraction.")
                    error_writer.writerow(row)
                    errorfile.flush()
                    continue

                # Get findings and clean them
                findings = findings_output.get("Findings", "")
                # Store findings
                row['Findings'] = findings

                # Write the row to the output CSV
                writer.writerow(row)
                outfile.flush()

    print(f"\nAll findings saved to {output_csv_path}")
    print(f"Errors saved to {error_csv_path}")

# Call the function to process the CSV file
process_csv_file(input_csv_path, output_csv_path)
