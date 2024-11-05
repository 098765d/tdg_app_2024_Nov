import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import pytz

def generate_predictions_and_save(model, X_test, y_test , df_test, target_col, optimal_cut_off, prog_code, test_cohort):
    """
    Generates GPA predictions, classifies at-risk students, and saves the results.

    Args:
    - model: Trained model to predict GPA.
    - X_test: Test features.
    - df_test: Test DataFrame.
    - target_col: The column for which predictions are being made.
    - optimal_cut_off: Threshold to classify 'At risk' students.
    - prog_code: The program code to name the output file.
    - test_cohort: Cohort information for the test set.

    Returns:
    - Modified df_test with predictions and risk classification.
    """

    # Model predictions (continuous and label)
    y_test_pred, y_test_label_pred = model.predict(X_test)
    df_test[f'{target_col}_prediction'] = np.round(y_test_pred, 3)  # Add predictions

    # Classify 'At risk' based on the optimal cut-off
    df_test['at_risk_prediction'] = df_test[f'{target_col}_prediction'] < optimal_cut_off

    # Drop the original target column from df_test
    # df_test=df_test.drop(columns=[target_col])

    # Create output directory if it doesn't exist
    output_dir = f'./{prog_code}/model_pred/'
    os.makedirs(output_dir, exist_ok=True)

    # Save predictions to an Excel file
    df_test_cleaned = df_test.dropna(thresh=int(0.7 * len(df_test)), axis=1)
    # rename the col 'SID' to 'pseduo_sid'
    df_test_cleaned.rename(columns={'SID': 'pseudo_sid'}, inplace=True)
    # Save the cleaned DataFrame to an Excel file
    excel_file_name = f"{prog_code}_predictions.xlsx"
    df_test_cleaned.to_excel(os.path.join(output_dir, excel_file_name), index=False)

    # Calculate evaluation metrics
    N = len(X_test)
    num_at_risk = np.sum(df_test['at_risk_prediction'])

    # Get current date in Hong Kong time zone
    hk_tz = pytz.timezone('Asia/Hong_Kong')
    current_date = datetime.now(hk_tz).strftime('%Y-%m-%d')

    # Create the result dictionary
    result = {
        "Program Code": prog_code,
        "Test Cohort": test_cohort,
        "Predicted GPA": target_col,
        "Test Cohort Size": N,
        "Num of At-Risk Prediction": num_at_risk,
        #"Date": current_date,
    }

    # Convert numpy types to native Python types
    result = {key: int(value) if isinstance(value, (np.integer, np.int64)) else value for key, value in result.items()}

    # Save result as JSON
    json_file_name = f"pred_summary.json"
    with open(os.path.join(output_dir, json_file_name), 'w') as f:
        json.dump(result, f, indent=4)

    return df_test_cleaned

# Example usage
# df_test_with_predictions = generate_predictions_and_save(model, X_test, df_test, target_col, optimal_cut_off, prog_code, test_cohort)
