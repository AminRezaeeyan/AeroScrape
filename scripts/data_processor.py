import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import logging
from scripts.config_loader import load_config # Assuming config_loader.py is in the same directory

logger = logging.getLogger(__name__)

def load_data(config):
    """Loads data from the path specified in config."""
    # Adjust path if running from Airflow context - this logic will be handled by DAG params or get_project_root in tasks
    data_path = config['data_path']
    # if not os.path.exists(data_path) and 'dags' in os.getcwd():
    #     project_root = os.path.dirname(os.getcwd())
    #     data_path = os.path.join(project_root, data_path)
    logger.info(f"Loading data from {data_path}")
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_path}. Please provide the correct path in config.")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(df, config, project_root_path):
    """
    Performs preprocessing and feature engineering.
    Returns X_train_p, X_test_p, y_reg_train, y_reg_test, y_cls_train, y_cls_test, preprocessor_path
    """
    logger.info("Starting data preprocessing...")

    # Make a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()

    # 1. Create target variable for classification
    df_processed[config['target_classification']] = (df_processed[config['target_regression']] > config['delay_threshold_minutes']).astype(int)
    logger.info(f"Created classification target '{config['target_classification']}'.")

    # 2. Feature Engineering: Extract minutes from 'scheduled_time'
    try:
        df_processed['scheduled_minute'] = pd.to_datetime(df_processed['scheduled_time'], format='%H:%M:%S', errors='raise').dt.minute
        logger.info("Extracted 'scheduled_minute' from 'scheduled_time'.")
    except Exception as e:
        logger.warning(f"Could not parse 'scheduled_time' to extract minutes: {e}. 'scheduled_minute' will be set to 0 or NaN.")
        df_processed['scheduled_minute'] = 0 # Or pd.NA, and handle with imputer if necessary

    # 3. Define features and targets
    X = df_processed[config['numerical_features'] + config['categorical_features']]
    y_reg = df_processed[config['target_regression']]
    y_cls = df_processed[config['target_classification']]

    missing_num_feats = [f for f in config['numerical_features'] if f not in X.columns]
    if missing_num_feats:
        logger.error(f"Missing numerical features after initial processing: {missing_num_feats}")
        raise ValueError(f"Missing numerical features: {missing_num_feats}")

    missing_cat_feats = [f for f in config['categorical_features'] if f not in X.columns]
    if missing_cat_feats:
        logger.error(f"Missing categorical features after initial processing: {missing_cat_feats}")
        raise ValueError(f"Missing categorical features: {missing_cat_feats}")

    # 4. Split data
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=config['test_size'], random_state=config['random_state']
    )
    logger.info(f"Data split into train and test sets. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    for col in config['numerical_features']:
        if X_train[col].dtype == 'object':
            try:
                X_train[col] = pd.to_numeric(X_train[col], errors='raise')
                X_test[col] = pd.to_numeric(X_test[col], errors='raise')
            except ValueError as e:
                logger.error(f"Could not convert numerical feature '{col}' to numeric: {e}. Check data quality.")
                raise

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, config['numerical_features']),
            ('cat', categorical_transformer, config['categorical_features'])
        ],
        remainder='passthrough'
    )
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    logger.info("Preprocessor fitted and data transformed.")

    model_output_dir_config = config['model_output_dir']
    # Construct absolute path for model_output_dir using project_root_path
    model_output_dir = os.path.join(project_root_path, model_output_dir_config)
    
    os.makedirs(model_output_dir, exist_ok=True)
    preprocessor_path = os.path.join(model_output_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Preprocessor saved to {preprocessor_path}")

    return X_train_processed, X_test_processed, y_reg_train, y_reg_test, y_cls_train, y_cls_test, preprocessor_path

if __name__ == '__main__':
    test_config_path = "../config/pipeline_config.yaml" 
    if not os.path.exists(test_config_path):
        test_config_path = "config/pipeline_config.yaml"
        
    test_config = load_config(test_config_path)
    
    # Determine project root for testing standalone
    # This mimics how get_project_root() would work if this script was in 'scripts' dir
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_test = os.path.dirname(current_file_dir) # Goes up one level from 'scripts'

    dummy_data_path_config = test_config['cleaned_data_path'] # Assuming processor works on cleaned data
    dummy_data_path_abs = os.path.join(project_root_for_test, dummy_data_path_config)

    if not os.path.exists(dummy_data_path_abs):
        print(f"Creating dummy data at {dummy_data_path_abs} for testing.")
        os.makedirs(os.path.dirname(dummy_data_path_abs), exist_ok=True)
        dummy_df_data = {
            'scheduled_time': ['06:50:00', '07:00:00', '08:00:00', '09:00:00', '10:00:00'],
            'airline': ['کاسپین', 'اطلس ایر', 'آتا', 'پويا', 'ایران ایر'],
            'destination_or_origin': ['مشهد', 'كيش', 'سیرجان', 'اصفهان', 'اهواز'],
            'aircraft': ['MD83', 'MD83', '737-700', 'EMB145', 'A319'],
            'airport': ['فرودگاه مهرآباد'] * 5,
            'flight_type': ['Domestic Departure'] * 5,
            'scheduled_day_of_week': ['Wednesday', 'Wednesday', 'Thursday', 'Thursday', 'Friday'],
            'is_weekend': [0,0,0,0,1],
            'is_Nourooz_4': [0]*5, 'is_Nourooz_13': [0]*5, 'Normal_holiday': [0]*5,
            'scheduled_season': ['Spring']*5,
            'scheduled_time_of_day': ['Morning']*5,
            'Scheduled_Hour_of_Day': [6,7,8,9,10],
            'delay_minutes': [31.0, 39.0, 10.0, 5.0, 65.0]
        }
        pd.DataFrame(dummy_df_data).to_csv(dummy_data_path_abs, index=False)
    
    try:
        # Adjust config's data_path to be absolute for the test
        test_config['data_path'] = dummy_data_path_abs 
        df_test = load_data(test_config)
        if not df_test.empty:
            X_train_p, X_test_p, y_reg_tr, y_reg_te, y_cls_tr, y_cls_te, prep_path = preprocess_data(df_test, test_config, project_root_for_test)
            logger.info(f"Test preprocessing successful. X_train_processed shape: {X_train_p.shape}")
            logger.info(f"Preprocessor saved at: {prep_path}")
    except Exception as e:
        logger.error(f"Error during data_processor.py test: {e}", exc_info=True)