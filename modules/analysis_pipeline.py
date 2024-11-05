
import sys
import os
import numpy as np

# Add the parent directory to sys.path to allow for module imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


import os
import json
from modules.process import extract_program_codes, load_data, filter_cohorts, filter_df_course, merge_dfs
from modules.model_train import prepare_train_test_data, train_and_generate_plots
from modules.model_predict import generate_predictions_and_save
from modules.generate_report import generate_report  # Import the generate_report function
from sklearn.metrics import mean_squared_error, accuracy_score

# Define constants
DATA_FOLDER_PATH = 'inter_data/'
FILTER_COEF = 0.8
BASE_FOLDER = '.'

def run_pipeline(mode='pred'):
    # Step 1: Extract and display program codes
    program_codes = extract_program_codes(DATA_FOLDER_PATH)
    # Initialize the PL_ID (if not existed)

    print("Program Codes Found:", program_codes)
    print("Total unique program codes:", len(program_codes))

    # Initialize a list to store combined data for all programs
    all_combined_data = []

    # Step 2: Process each program code
    for prog_code in program_codes:
        print('-------------------------------')
        print(f"Processing program code: {prog_code}")

        # Step 2.1: Load data for the selected program code
        df_course, df_grades = load_data(prog_code, DATA_FOLDER_PATH)

        # Step 2.2: Filter cohorts and course data
        train_cohorts, test_cohort, target_col, filtered_df_grades = filter_cohorts(df_grades,mode=mode)
        filtered_df_course = filter_df_course(filtered_df_grades, df_course, FILTER_COEF)

        # Step 2.3: Merge filtered data for training and testing
        merged_df, df_train, df_test = merge_dfs(filtered_df_grades, filtered_df_course)
        # calculate the correlation between courses and target col in df_train
        course_codes = filtered_df_course.columns.drop('SID').tolist()
        course_target_corr = df_train[course_codes].corrwith(df_train[target_col])
        print('course_target_corr: ', course_target_corr)
        # output the courses with correlation >= 0.5 and the correlation value in a dict named high_correlation_courses
        high_correlation_courses = {course: corr for course, corr in course_target_corr.items() if abs(corr) >= 0.5}
        # keep 3 decimal places and sort the dict by correlation value from high to low
        high_correlation_courses = {course: round(corr, 3) for course, corr in high_correlation_courses.items()}
        high_correlation_courses = dict(sorted(high_correlation_courses.items(), key=lambda x: abs(x[1]), reverse=True))
        print('high_correlation_courses: ', high_correlation_courses)


        # Step 3: Prepare train and test data
        X_train, y_train, X_test, y_test, scaler, imputer = prepare_train_test_data(
            df_train=df_train,
            df_test=df_test,
            target_col=target_col
        )
        print('y_test: ', type(y_test))

        # Step 4: Train the model and generate evaluation plots
        model, optimal_cut_off = train_and_generate_plots(
            X_train=X_train,
            y_train=y_train,
            prog_code=prog_code,
            cohorts=train_cohorts
        )
        print(f"Model trained for {prog_code} with optimal cut-off: {optimal_cut_off}")
        with open(os.path.join(BASE_FOLDER, prog_code, 'model_eval', 'corr_course.json'), 'w') as f:
            json.dump(high_correlation_courses, f, indent=4)

        # Step 5: Generate predictions and save results
        df_test_with_predictions = generate_predictions_and_save(
            model, X_test,y_test,df_test, target_col, optimal_cut_off, prog_code, test_cohort)
        print(f"Predictions saved for {prog_code}\n")

        # Step 6: Generate report (optional)
        generate_report(prog_code, BASE_FOLDER)
        print(f"Report generated for {prog_code}\n")

        # Step 7: Extract data from eval_summary.json and pred_summary.json
        eval_file_path = os.path.join(BASE_FOLDER, prog_code, 'model_eval', 'eval_summary.json')
        pred_file_path = os.path.join(BASE_FOLDER, prog_code, 'model_pred', 'pred_summary.json')


        
        combined_data = {
            'program_code': prog_code,
            'eval_summary': None,
            'pred_summary': None
        }

        # Load eval_summary.json
        if os.path.exists(eval_file_path):
            with open(eval_file_path, 'r', encoding='utf-8') as eval_file:
                eval_data = json.load(eval_file)
                combined_data['eval_summary'] = eval_data
        else:
            print(f"Warning: {eval_file_path} not found.")

        # Load pred_summary.json
        if os.path.exists(pred_file_path):
            with open(pred_file_path, 'r', encoding='utf-8') as pred_file:
                pred_data = json.load(pred_file)
                combined_data['pred_summary'] = pred_data
        else:
            print(f"Warning: {pred_file_path} not found.")

        # Add the combined data for this program to the list
        all_combined_data.append(combined_data)

        # Assuming you are in the 'val' mode of operation and already have df_test_with_predictions loaded with the necessary columns

        if mode == 'val':
            # Handle NaN values in the dataset
            # Dropping rows where either the target or the prediction contains NaN
            df_test_with_predictions.dropna(subset=[target_col, target_col+'_prediction'], inplace=True)

            # Calculate MSE and accuracy after ensuring there are no NaN values
            mse = mean_squared_error(df_test_with_predictions[target_col], df_test_with_predictions[target_col+'_prediction'])
            actual_label = df_test_with_predictions[target_col] <= 2.5  # True for at-risk
            pred_label = df_test_with_predictions[target_col+'_prediction'] <= optimal_cut_off  # True for predicted at-risk
            acc = accuracy_score(actual_label, pred_label)

            # Calculate the number of at-risk and missed detections
            num_at_risk = actual_label.sum()
            num_missed_detection = ((actual_label == True) & (pred_label == False)).sum()

            # Prepare the result dictionary, ensuring all data are in serializable format
            result = {
                "Program Code": str(prog_code),
                "Train Cohorts": [int(x) for x in train_cohorts],
                "Test Cohort": int(test_cohort),
                "Train Cohort Size": int(len(df_train)),
                "Test Cohort Size": int(len(df_test)),
                "MSE": float(mse),
                "Accuracy": float(acc),
                "At-Risk Count": int(num_at_risk),
                "Missed Detections": int(num_missed_detection)
            }

            # Format the result as text and save to a text file
            result_text = (
                f"Program Code: {result['Program Code']}\n"
                f"Train Cohorts: {result['Train Cohorts']}\n"
                f"Test Cohort: {result['Test Cohort']}\n"
                f"Train Cohort Size: {result['Train Cohort Size']}\n"
                f"Test Cohort Size: {result['Test Cohort Size']}\n"
                f"MSE: {result['MSE']:.4f}\n"
                f"Accuracy: {result['Accuracy']:.2f}\n"
                f"At-Risk Count: {result['At-Risk Count']}\n"
                f"Missed Detections: {result['Missed Detections']}\n"
                "----------------------------------------\n"
            )

            # Append the formatted result to the file
            with open('validation_results.txt', 'a') as file:
                file.write(result_text)


    # After processing all programs, write the accumulated data to model_log.json
    output_file_path = os.path.join(BASE_FOLDER, 'model_log.json')
    
    try:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(all_combined_data, output_file, ensure_ascii=False, indent=4)
        print(f"Combined data saved to {output_file_path}")
    except Exception as e:
        print(f"Error writing to {output_file_path}: {e}")
    
    # Assuming you are in the 'val' mode of operation

    


# Entry point of the script
if __name__ == "__main__":
    run_pipeline()