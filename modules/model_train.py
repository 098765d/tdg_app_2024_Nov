# model_train.py
from modules.dualPredictor.df_preprocess import data_preprocessing, check_df_forms
from modules.dualPredictor import DualModel
from modules.dualPredictor.model_plot import plot_scatter, plot_cm, plot_feature_coefficients


def prepare_train_test_data(df_train, df_test, target_col, id_col='SID', drop_cols=None, scaler=None, imputer=None):
    """
    Preprocesses the training and testing datasets, applies scaling and imputation,
    and ensures that the training and testing feature sets have consistent forms.

    Args:
        df_train (pd.DataFrame): The training dataset.
        df_test (pd.DataFrame): The testing dataset.
        target_col (str): The column name for the target variable.
        id_col (str): The identifier column (default is 'SID').
        drop_cols (list): A list of columns to drop from the datasets (default is ['Cohort', 'Yr']).
        scaler (sklearn.preprocessing): Optional scaler to use. If None, a new scaler will be fit on the training data.
        imputer (sklearn.impute): Optional imputer to use. If None, a new imputer will be fit on the training data.

    Returns:
        X_train (pd.DataFrame): Preprocessed training features.
        y_train (pd.Series): Target values for training.
        X_test (pd.DataFrame): Preprocessed testing features.
        y_test (pd.Series): Target values for testing.
        scaler: The scaler used for preprocessing.
        imputer: The imputer used for preprocessing.
    """
    if drop_cols is None:
        drop_cols = ['Cohort', 'Yr']  # Default drop columns

    # Step 1: Data preprocessing for training data
    X_train, y_train, scaler, imputer = data_preprocessing(
        df=df_train,
        target_col=target_col,
        id_col=id_col,
        drop_cols=drop_cols,
        scaler=scaler,
        imputer=imputer
    )

    # Step 2: Data preprocessing for testing data
    X_test, y_test, scaler, imputer = data_preprocessing(
        df=df_test,
        target_col=target_col,
        id_col=id_col,
        drop_cols=drop_cols,
        scaler=scaler,
        imputer=imputer
    )

    # Step 3: Ensure X_train and X_test have the same form
    X_train, X_test = check_df_forms(X_train, X_test)

    return X_train, y_train, X_test, y_test, scaler, imputer

# Example usage:
'''
X_train, y_train, X_test, y_test, scaler, imputer = prepare_train_test_data(
    df_train=df_train,
    df_test=df_test,
    target_col=target_col)


# Verify if preprocessing was successful
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

'''

import numpy as np
from modules.dualPredictor import DualModel
from modules.dualPredictor.model_plot import plot_scatter, plot_cm, plot_feature_coefficients
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, accuracy_score
import json
from datetime import datetime
import pytz

def train_and_generate_plots(X_train, y_train, prog_code, cohorts, model_type='lasso', metric='youden_index', default_cut_off=2.5):
    """
    Trains a LASSO model, generates plots, and calculates evaluation metrics.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target values.
        prog_code (str): Program code for the model.
        cohorts (list): List of cohorts for training.
        model_type (str): Model type, default is 'lasso'.
        metric (str): Evaluation metric, default is 'youden_index'.
        default_cut_off (float): Cut-off value for classification.

    Returns:
        model (DualModel): Trained model.
        optimal_cut_off (float): Optimal cut-off value from the model.
    """

    # Directory setup
    result_dir = f'./{prog_code}/model_eval/'
    img_dir = os.path.join(result_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)

    # Train the model
    model = DualModel(model_type, metric, default_cut_off)
    model.fit(X_train, y_train)
    optimal_cut_off = model.optimal_cut_off
    print(f'Optimal cut-off = {optimal_cut_off}')

    # Training set predictions and true labels
    y_train_pred, y_train_label_pred = model.predict(X_train)
    y_train_label_true = (y_train < default_cut_off).astype(int)

    # Save scatter plot
    scatter_fig_path = os.path.join(img_dir, 'scatter_plot.png')
    plot_scatter(y_train_pred, y_train).savefig(scatter_fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save confusion matrix
    cm_fig_path = os.path.join(img_dir, 'confusion_matrix.png')
    plot_cm(y_train_label_true, y_train_label_pred)
    plt.savefig(cm_fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save feature importance plot
    feature_imp_fig_path = os.path.join(img_dir, 'feature_importance.png')
    plot_feature_coefficients(model.coef_, model.feature_names_in_)
    plt.savefig(feature_imp_fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Metrics calculation
    N = len(X_train)
    mse = mean_squared_error(y_train, y_train_pred)
    num_at_risk = np.sum(y_train_label_true == 1)
    miss_detect = np.sum((y_train_label_true == 1) & (y_train_label_pred == 0))
    false_alarm = np.sum((y_train_label_true == 0) & (y_train_label_pred == 1))
    accuracy = round(accuracy_score(y_train_label_true, y_train_label_pred), 3)

    # Determine target column
    prog_duration = int(prog_code[1])
    target_column = 'Y3.GPA' if prog_duration >= 3 else 'Gd.GPA'

    # Current date in Hong Kong time zone
    hk_tz = pytz.timezone('Asia/Hong_Kong')
    current_date = datetime.now(hk_tz).strftime('%Y-%m-%d')

    # Create result dictionary
    result = {
        "Program Code": prog_code,
        "Target Column": target_column,
        "Train Cohorts": ' '.join([str(cohort) for cohort in cohorts]),
        "Cohort Size": N,
        "Num of At-Risk": num_at_risk,
        "Miss Detect": miss_detect,
        "False Alarm": false_alarm,
        "Optimal Cut-Off": round(optimal_cut_off, 2),
        "Accuracy (%)": round(accuracy * 100, 2),
        "MSE": round(mse, 3),
        # "Date": current_date,
    }
    # Convert numpy data types to native Python types
    result = {key: int(value) if isinstance(value, (np.integer, np.int64)) else value for key, value in result.items()}

    # Save result as JSON
    with open(os.path.join(result_dir, 'eval_summary.json'), 'w') as f:
        json.dump(result, f, indent=4)

    return model, optimal_cut_off

# Example usage
# model, optimal_cut_off = train_and_generate_plots(X_train, y_train, prog_code)

