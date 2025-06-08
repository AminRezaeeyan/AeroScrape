# scripts/data_cleaner.py
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def clean_and_engineer_features(raw_data_path, cleaned_data_path):
    """
    Loads raw data, cleans it, engineers features, and saves the result.
    This function encapsulates the logic from your provided script.
    """
    logger.info(f"Starting data cleaning and feature engineering from: {raw_data_path}")
    
    # 1. Load Data
    try:
        df = pd.read_csv(raw_data_path)
        logger.info(f"Successfully loaded raw data. Shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Raw data file not found at {raw_data_path}. Halting process.")
        raise

    # 2. Filtering
    logger.info("Filtering flights...")
    df = df[(df['flight_type'] != 'Domestic Arrival') & (df['flight_type'] != 'International Arrival') & (df['flight_type'] != 'International Departure')]
    df = df[df['status'] == 'پرواز كرد']
    logger.info(f"Shape after filtering: {df.shape}")

    # 3. Feature Engineering
    logger.info("Engineering features...")
    df['scheduled'] = pd.to_datetime(df['scheduled_time'], errors='coerce')
    df['actual'] = pd.to_datetime(df['actual_time'], errors='coerce')

    df['scheduled_datetime'] = pd.to_datetime(df['scheduled'], errors='coerce')
    df['actual_datetime'] = pd.to_datetime(df['actual'], errors='coerce')

    df['delay'] = df['actual_datetime'] - df['scheduled_datetime']
    df['delay_minutes'] = df['delay'].dt.total_seconds() / 60

    # Date/Time components
    df['scheduled_day_of_week'] = df['scheduled_datetime'].dt.day_name()
    df['is_weekend'] = df['scheduled_day_of_week'].isin(['Thursday', 'Friday']).astype(int)
    
    # Holiday features (for 2025 as in your script)
    df_valid_dates = df[df['scheduled_datetime'].notna()].copy()
    if not df_valid_dates.empty:
        nourooz_4_dates = [(3, 21), (3, 22), (3, 23), (3, 24)]
        df_valid_dates['is_Nourooz_4'] = df_valid_dates['scheduled_datetime'].apply(lambda x: 1 if (x.month, x.day) in nourooz_4_dates else 0)
        df_valid_dates['is_Nourooz_13'] = df_valid_dates['scheduled_datetime'].apply(lambda x: 1 if (x.month == 3 and x.day >= 21) or (x.month == 4 and x.day <= 2) else 0)
        
        other_fixed_holidays = [(3, 20), (2, 11), (6, 4), (6, 5)]
        def check_normal_holiday(row):
            if row['is_Nourooz_13'] == 1: return 0
            if (row['scheduled_datetime'].month, row['scheduled_datetime'].day) in other_fixed_holidays: return 1
            return 0
        df_valid_dates['Normal_holiday'] = df_valid_dates.apply(check_normal_holiday, axis=1)

        df = df.merge(df_valid_dates[['is_Nourooz_4', 'is_Nourooz_13', 'Normal_holiday']], left_index=True, right_index=True, how='left')
        df[['is_Nourooz_4', 'is_Nourooz_13', 'Normal_holiday']] = df[['is_Nourooz_4', 'is_Nourooz_13', 'Normal_holiday']].fillna(0).astype(int)
    else:
        df['is_Nourooz_4'], df['is_Nourooz_13'], df['Normal_holiday'] = 0, 0, 0
    
    # Season and Time of Day
    def get_season(dt):
        if pd.isna(dt): return "Unknown"
        m, d = dt.month, dt.day
        if (m == 3 and d >= 21) or m in [4, 5] or (m == 6 and d <= 20): return "Spring"
        if (m == 6 and d >= 21) or m in [7, 8] or (m == 9 and d <= 22): return "Summer"
        if (m == 9 and d >= 23) or m in [10, 11] or (m == 12 and d <= 21): return "Autumn"
        return "Winter"
    
    def get_time_of_day(dt):
        if pd.isna(dt): return "Unknown"
        h = dt.hour
        if 5 <= h <= 11: return "Morning"
        elif 12 <= h <= 16: return "Afternoon"
        elif 17 <= h <= 20: return "Evening"
        return "Night"

    df['scheduled_season'] = df['scheduled_datetime'].apply(get_season)
    df['scheduled_time_of_day'] = df['scheduled_datetime'].apply(get_time_of_day)

    # Derived delay features
    df['Early_OnTime_Late_Indicator'] = df['delay_minutes'].apply(lambda x: "Early" if x < 0 else ("On Time" if x <= 15 else "Late"))
    df['Actual_Day_Matches_Scheduled_Day'] = (df['scheduled_datetime'].dt.date == df['actual_datetime'].dt.date).astype(int)
    df['Scheduled_Hour_of_Day'] = df['scheduled_datetime'].dt.hour
    
    # Handle NaNs in the target column before saving
    initial_rows = len(df)
    df.dropna(subset=['delay_minutes'], inplace=True)
    logger.info(f"Dropped {initial_rows - len(df)} rows with missing 'delay_minutes'.")
    
    # Reset index before saving
    df.reset_index(drop=True, inplace=True)
    
    logger.info("Feature engineering complete.")
    
    # 4. Save the cleaned and engineered DataFrame
    try:
        df.to_csv(cleaned_data_path, index=False, encoding='utf-8-sig')
        logger.info(f"DataFrame successfully saved to CSV: {cleaned_data_path}")
    except Exception as e:
        logger.error(f"Error saving cleaned data file: {e}")
        raise

    return cleaned_data_path