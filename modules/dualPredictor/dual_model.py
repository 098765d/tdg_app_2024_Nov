# dualPredictor/dual_model.py

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from sklearn.metrics import confusion_matrix, balanced_accuracy_score,fbeta_score
import numpy as np

'''
This module, dualPredictor/dual_model.py, defines the DualModel class, which integrates regression and classification capabilities into a single model framework. 
The class is built on scikit-learn's BaseEstimator, allowing it to be used as a typical sklearn model with additional functionality.

Class DualModel:
    - A flexible model that can use Lasso, Ridge, or Ordinary Least Squares (OLS) regression based on user selection, and automatically tunes a decision threshold for binary classification.
    
    Parameters:
        model_type (str): Specifies the type of regression model ('lasso', 'ridge', 'ols').
        metric (str): The metric used to optimize the classification cutoff ('youden_index', 'f1_score', 'f2_score').
        default_cut_off (float): The initial threshold for converting regression outputs to binary labels.

    Methods:
        __init__(self, model_type='lasso', metric='youden_index', default_cut_off=0.5):
            - Initializes the model with the specified regression type and metrics.
        
        fit(self, X, y):
            - Fits the model to the data and tunes the cutoff for binary classification based on the provided metric.
            - Parameters:
                X (array-like, shape (n_samples, n_features)): Training data.
                y (array-like, shape (n_samples,)): Target values.
            - Returns:
                self: Fitted estimator.

        predict(self, X):
            - Predicts using the fitted model. Returns both continuous and binary predictions.
            - Parameters:
                X (array-like, shape (n_samples, n_features)): Data to predict.
            - Returns:
                tuple: Tuple containing continuous predictions and binary predictions.

    Attributes:
        metrics_:
            - Returns the metric values evaluated at each cutoff during tuning.
        
        cutoffs_:
            - Returns the cutoff values evaluated during tuning.
        
        alpha_:
            - Returns the regularization strength parameter (alpha) for Lasso and Ridge.
        
        coef_:
            - Returns the coefficients of the regression model.
        
        intercept_:
            - Returns the intercept of the regression model.
        
        feature_names_in_:
            - Returns the input feature names.

Usage:
The DualModel is suitable for scenarios where one needs the flexibility to perform both regression and classification tasks within the same model framework, 
leveraging the strengths of regularization and threshold optimization.
'''

class DualModel(BaseEstimator, RegressorMixin):
    def __init__(self, model_type='lasso', metric='youden_index', default_cut_off=0.5):
        self.model_type = model_type
        self.metric = metric
        self.default_cut_off = default_cut_off
        self.model = None
        self.optimal_cut_off = default_cut_off  # Default to using the default cut-off
        self.y_label_true_ = None
        self.metrics = None
        self.cutoffs = None

        if model_type == 'lasso':
            self.model = LassoCV(cv=5)
        elif model_type == 'ridge':
            self.model = RidgeCV(cv=5)
        elif model_type == 'ols':
            self.model = LinearRegression()
            self.model.alpha_ = 0  # Set alpha to 0 for OLS
        else:
            raise ValueError("Unsupported model type. Choose 'lasso', 'ridge', or 'ols'.")

    def fit(self, X, y):
        self.model.fit(X, y)

        # Create binary labels based on the default cut-off
        y_label_true = (y < self.default_cut_off).astype(int)
        self.y_label_true_ = y_label_true

        # Error handling for cases where the default cut-off is too high
        if self.default_cut_off >= np.max(y):
            raise ValueError("The default cut-off must be smaller than the maximum value of y.")

        # Check if there are any y values less than the default cut-off
        if np.all(y >= self.default_cut_off):
            # If no y values are less than the default cut-off, skip tuning and use default cut-off
            print("No y values less than the default cut-off. Using default cut-off for classification.")
            self.optimal_cut_off = self.default_cut_off
        else:
            # Tune the optimal cut-off
            cut_offs = np.linspace(self.default_cut_off, max(y), 55)
            metrics = []

            for cut_off in cut_offs:
                y_pred = (self.model.predict(X) < cut_off).astype(int)
                if self.metric == 'f1_score':
                    metric_value = fbeta_score(y_label_true, y_pred, beta=1)
                elif self.metric == 'f2_score':
                    metric_value = fbeta_score(y_label_true, y_pred, beta=2)
                elif self.metric == 'youden_index':
                    metric_value = balanced_accuracy_score(y_label_true, y_pred, adjusted=True)
                else:
                    raise ValueError("Unsupported metric. Choose 'f1_score', 'f2_score', or 'youden_index'.")

                metrics.append(metric_value)

            max_metric = max(metrics)
            max_indices = [i for i, x in enumerate(metrics) if x == max_metric]
            middle_index = max_indices[len(max_indices) // 2]
            self.optimal_cut_off = cut_offs[middle_index]
            self.metrics = metrics
            self.cutoffs = cut_offs

        return self

    def predict(self, X):
        grade_predictions = self.model.predict(X)
        class_predictions = (grade_predictions < self.optimal_cut_off).astype(int)
        return grade_predictions, class_predictions

    @property
    def metrics_(self):
        return self.metrics

    @property
    def cutoffs_(self):
        return self.cutoffs

    @property
    def alpha_(self):
        return self.model.alpha_

    @property
    def coef_(self):
        return self.model.coef_

    @property
    def intercept_(self):
        return self.model.intercept_

    @property
    def feature_names_in_(self):
        return getattr(self.model, 'feature_names_in_', None)