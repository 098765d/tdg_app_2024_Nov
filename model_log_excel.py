import json
import pandas as pd

# Load the JSON data
with open('model_log.json', 'r') as f:
    model_log = json.load(f)

# Initialize lists to store the summary data
data = []

# Loop through each program in the model log
for entry in model_log:
    program_code = entry['program_code']
    
    # Extract evaluation summary
    eval_summary = entry.get('eval_summary', {})
    pred_summary = entry.get('pred_summary', {})
    
    # Prepare a dictionary of the required information
    summary = {
        'Program Code': program_code,
        'Train Cohorts': eval_summary.get('Train Cohorts', ''),
        'Cohort Size (Train)': eval_summary.get('Cohort Size', 0),
        'Num of At-Risk (Train)': eval_summary.get('Num of At-Risk', 0),
        'Miss Detect (Train)': eval_summary.get('Miss Detect', 0),
        'False Alarm (Train)': eval_summary.get('False Alarm', 0),
        'Optimal Cut-Off (Train)': eval_summary.get('Optimal Cut-Off', ''),
        'Accuracy (%) (Train)': eval_summary.get('Accuracy (%)', 0),
        'MSE (Train)': eval_summary.get('MSE', 0),
        
        'Test Cohort': pred_summary.get('Test Cohort', ''),
        'Test Cohort Size': pred_summary.get('Test Cohort Size', 0),
        'Num of At-Risk Prediction (Test)': pred_summary.get('Num of At-Risk Prediction', 0),
    }
    
    # Append to the data list
    data.append(summary)

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(data)

# Write the DataFrame to an Excel file
output_file = 'program_summary.xlsx'
df.to_excel(output_file, index=False)

print(f"Summary has been saved to {output_file}")

import os

print("Current working directory:", os.getcwd())

