import pandas as pd

# Path to the uploaded text file
file_path = 'validation_results.txt'

# Initialize an empty list to store each block of data as a dictionary
results = []

# Read the file and parse the data
with open(file_path, 'r') as file:
    current_result = {}
    for line in file:
        line = line.strip()
        if line == "----------------------------------------":
            # End of a block of results, add it to the list and reset the dictionary
            if current_result:
                results.append(current_result)
                current_result = {}
        elif line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            # Handle list values and numerical conversions
            if '[' in value:
                value = eval(value)  # Convert string list to actual list
            elif value.replace('.', '', 1).isdigit():
                if '.' in value:
                    value = float(value)  # Convert to float
                else:
                    value = int(value)  # Convert to int
            current_result[key] = value

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(results)

# Display the DataFrame to the user
df.to_excel('validation_results.xlsx', index=False)
