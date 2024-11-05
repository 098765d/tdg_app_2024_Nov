import os
import re
import pandas as pd
import numpy as np

def extract_program_codes(data_folder_path):
    """
    Extracts unique program codes (including majors, if present) from the filenames in the specified folder.

    Args:
    data_folder_path (str): The path to the folder containing the files.

    Returns:
    list: A list of unique program codes.
    """

    # Step 2: List all files in the folder
    file_names = os.listdir(data_folder_path)

    # Step 3: Define a pattern to match program codes with or without a major
    pattern = r'(A[0-9A-Z]+)(?:-([A-Za-z]+))?_'

    # Step 4: Extract program codes (including major, if present) from the file names using the pattern
    program_codes = [
        f"{match.group(1)}{'-' + match.group(2) if match.group(2) else ''}"
        for file_name in file_names
        if (match := re.search(pattern, file_name))
    ]

    # Step 5: Remove duplicates by converting the list to a set and back to a list
    unique_program_codes = list(set(program_codes))

    # Return the list of unique program codes
    return unique_program_codes

# Example usage:
# data_folder_path = '/content/drive/MyDrive/eduhk_RA/TDG_Project/intake_2022_analysis/prog_data'
# program_codes = extract_program_codes(data_folder_path)




# Function to load the course and grades data for the selected program code
def load_data(selected_code, data_folder_path):
    course_file = None
    grades_file = None

    # List all files in the data folder
    file_names = os.listdir(data_folder_path)

    def rename_sid_column(df):
        # Define the regular expression pattern for exactly 8 alphanumeric characters
        pattern = r'^[A-Z0-9]{8}$'

        # Iterate over all columns in the DataFrame
        for col in df.columns:
            first_value = str(df[col].iloc[0]).strip()  # Get the first value, convert to string, and strip spaces
            if re.match(pattern, first_value):
                # If the first value matches the pattern, rename the column to 'SID'
                df.rename(columns={col: 'SID'}, inplace=True)
                print(f"Renamed column '{col}' to 'SID'")
                break  # Exit loop once the column is renamed
        return df  # Return the DataFrame after renaming

    def remove_course_grade_columns(df):
        # Define a regular expression pattern for course codes like 'ENG1324' or 'ENG1327.LIT1040' or multiple codes like 'CLE1168.CLE1210.CLE1241.CLE1250'
        course_grade_pattern = r'^([A-Z]{3}\d{4})(\.[A-Z]{3}\d{4})*$'

        # Get a list of columns to drop by checking if they match the pattern
        cols_to_drop = [col for col in df.columns if re.match(course_grade_pattern, col)]

        # Drop the detected columns from the dataframe
        df_cleaned = df.drop(columns=cols_to_drop)

        # Print the columns that were removed for reference
        print(f"Removed course grade columns: {cols_to_drop}")

        return df_cleaned

    # Initialize empty DataFrames in case files are not found
    df_course = pd.DataFrame()
    df_grades = pd.DataFrame()

    # Search for corresponding files for the selected program code
    for file_name in file_names:
        # Check if the file corresponds to the course or merged grades file
        if f"CourseGrade_{selected_code}" in file_name:
            course_file = file_name
            try:
                df_course = pd.read_csv(os.path.join(data_folder_path, course_file), encoding='utf-8')
            except UnicodeDecodeError:
                # If utf-8 encoding fails, try 'ISO-8859-1'
                df_course = pd.read_csv(os.path.join(data_folder_path, course_file), encoding='ISO-8859-1')

            df_course = rename_sid_column(df_course)

            # Define the grade-to-numerical mapping based on the provided scale and operational grade interpretations
            grade_mapping = {
                'A+': 4.33, 'A': 4.00, 'A-': 3.67,
                'B+': 3.33, 'B': 3.00, 'B-': 2.67,
                'C+': 2.33, 'C': 2.00, 'C-': 1.67,
                'D': 1.00, 'F': 0.00,
                'DN': 3.67, 'CR': 2.67, 'PS': 2.33,
                'FL': 0.00, 'IP': 0.00, 'YC': 2.33,
                'YI': 0.00, 'W': 0.00, 'PND': 0.00
            }

            # Convert the grades using the mapping; default to 0 for unknown grades
            df_course['Final.Grade'] = df_course['Final.Grade'].map(grade_mapping).fillna(0)
            print(f"Loaded course data from: {course_file}")

        elif f"Merged_{selected_code}" in file_name:
            grades_file = file_name
            try:
                df_grades = pd.read_csv(os.path.join(data_folder_path, grades_file), encoding='utf-8')
            except UnicodeDecodeError:
                # If utf-8 encoding fails, try 'ISO-8859-1'
                df_grades = pd.read_csv(os.path.join(data_folder_path, grades_file), encoding='ISO-8859-1')

            df_grades = remove_course_grade_columns(df_grades)
            df_grades = rename_sid_column(df_grades)
            print(f"Loaded grades data from: {grades_file}")

    return df_course, df_grades

# Load the data for the initial default selected code
# df_course, df_grades = load_data(prog_code, data_folder_path)
# df_grades.info()

import pandas as pd
import re

def filter_cohorts(df_grades,mode='pred'): # mode = 'pred' or 'val'
    def get_target_col(df_grades):
        # Determine whether the program is 3+ years or 2 years based on the GPA column
        if 'Y3.GPA' in df_grades.columns:
            target_col = 'Y3.GPA'  # Program with more than 3 years
        else:
            target_col = 'Gd.GPA'  # 2-year program
        return target_col

    def drop_columns_after_year(df, year_threshold, target_col):
        # Define a pattern to match columns representing years beyond the threshold
        pattern = r'Y([3-9]\d?|S\d|C\.)' if year_threshold == 2 else r'Y([2-9]\d?|S\d|C\.)'

        # Drop columns that match the pattern for years beyond the threshold
        cols_to_drop = [col for col in df.columns if re.search(pattern, col) and col != target_col]

        # Drop the columns and return the filtered DataFrame
        df_dropped = df.drop(columns=cols_to_drop, errors='ignore')
        return df_dropped

    def drop_columns_with_keywords(df, target_col, keywords):
        # Drop columns that contain specific keywords but not the target column
        cols_to_drop = [col for col in df.columns if any(keyword.lower() in col.lower() for keyword in keywords) and col != target_col]

        # Drop the columns and return the filtered DataFrame
        df_dropped = df.drop(columns=cols_to_drop, errors='ignore')
        return df_dropped

    # Get the target column (either Y3.GPA or Gd.GPA)
    target_col = get_target_col(df_grades)

    if mode == 'pred':
        # Filter the rows where the target column is not NaN
        filtered_df = df_grades[df_grades[target_col].notna()]
        # Determine the max training cohort and the cohorts to filter
        max_train_cohort = filtered_df['Cohort'].max()

        if target_col=='Y3.GPA':
            # Filter the DataFrame where 'Y2.GPA' is not None (or not null)
            test_cohort = df_grades[df_grades['Y2.GPA'].notna()]['Cohort'].max()
        else:
            test_cohort = max_train_cohort + 1
        train_cohorts = [max_train_cohort - 2, max_train_cohort - 1, max_train_cohort]
        filter_cohorts = [max_train_cohort - 2, max_train_cohort - 1, max_train_cohort, test_cohort]

    elif mode =='val':
        # Filter the rows where the target column is not NaN
        filtered_df = df_grades[df_grades[target_col].notna()]
        # Determine the max training cohort and the cohorts to filter
        max_val_cohort = filtered_df['Cohort'].max()
        second_max_cohort = np.sort(filtered_df['Cohort'].unique())[-2]
        third_max_cohort = np.sort(filtered_df['Cohort'].unique())[-3]
        fourth_max_cohort = np.sort(filtered_df['Cohort'].unique())[-4]
        fifth_max_cohort = np.sort(filtered_df['Cohort'].unique())[-5]

        if target_col=='Y3.GPA':
            test_cohort = max_val_cohort
        else:
            test_cohort = max_val_cohort
        
        train_cohorts = [fourth_max_cohort,third_max_cohort,second_max_cohort]
        filter_cohorts = [fourth_max_cohort,third_max_cohort,second_max_cohort, test_cohort]
        print('filter_cohorts:',filter_cohorts)


    # Filter the DataFrame based on the cohort filter
    filter_df_grades = df_grades[df_grades['Cohort'].isin(filter_cohorts)]

    # Drop columns based on program duration (detected by target_col)
    if target_col == 'Y3.GPA':
        # Program longer than 3 years: Drop columns after the 2nd year (Y3 and onwards)
        filter_df_grades = drop_columns_after_year(filter_df_grades, 2, target_col)
    else:
        # 2-year program: Drop columns after the 1st year (Y2 and onwards)
        filter_df_grades = drop_columns_after_year(filter_df_grades, 1, target_col)

    # Drop columns that contain the keywords "risk" and "Gd" except for the target column
    keywords_to_drop = ['risk', 'Gd']
    filter_df_grades = drop_columns_with_keywords(filter_df_grades, target_col, keywords_to_drop)

    return train_cohorts, test_cohort, target_col, filter_df_grades

# Example usage
# train_cohorts, test_cohort, target_col, filtered_df_grades = filter_cohorts(df_grades)


def filter_df_course(filter_df_grades, df_course,coef):
    # Step 1: Filter the df_course to keep only rows where SID matches the filter_df_grades['SID']
    filtered_df_course = df_course[df_course['SID'].isin(filter_df_grades['SID'])].copy()

    # Step 2: Pivot the filtered DataFrame
    # We want 'SID' as the index, 'Course.Code' as the columns, and 'Final.Grade' as the values
    pivoted_df_course = filtered_df_course.pivot_table(
        index='SID',
        columns='Course.Code',
        values='Final.Grade',
        aggfunc='first'  # Since we expect unique values for each SID and Course.Code combination
    )

    # Step 3: Filter columns with at least 80% non-null values
    threshold = coef * len(pivoted_df_course)  # Calculate the 80% threshold for non-null values
    filtered_df_course = pivoted_df_course.dropna(thresh=threshold, axis=1)

    # Step 4: Reset the index to bring 'SID' back as a column
    filtered_df_course.reset_index(inplace=True)

    return filtered_df_course

# Example usage
# filtered_df_course = filter_df_course(filtered_df_grades, df_course,0.8)

import pandas as pd

def merge_dfs(filtered_df_grades, filtered_df_course):
    # Perform a left join on the SID column
    merged_df = pd.merge(filtered_df_grades, filtered_df_course, on='SID', how='left')

    # Get the maximum value of the Cohort column
    max_cohort = merged_df['Cohort'].max()
    threshold = 0.5 * len(merged_df)
    merged_df = merged_df.dropna(thresh=threshold, axis=1)

    # Split into df_train (Cohort != max_cohort) and df_test (Cohort == max_cohort)
    df_test = merged_df[merged_df['Cohort'] == max_cohort].copy()
    df_train = merged_df[merged_df['Cohort'] != max_cohort].copy()

    return merged_df, df_train, df_test

# Example usage
# merged_df, df_train, df_test = merge_dfs(filtered_df_grades, filtered_df_course)
