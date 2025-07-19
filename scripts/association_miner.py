import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import logging

logger = logging.getLogger(__name__)

def find_association_rules(data_path, output_path, min_support=0.01, min_threshold=1) -> str:
    """
    Finds association rules in the flight data using FP-Growth.
    """
    logger.info(f"Starting association rule mining from: {data_path}")

    try: 
        df = pd.read_csv(data_path)
        logger.info(f"Successfully loaded data. Shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_path}, Halting Process")

    # Feature selection
    df_assoc = df[['airline', 'destination_or_origin', 'aircraft', 'airport', 'scheduled_day_of_week', 'scheduled_season', 'scheduled_time_of_day', 'Early_OnTime_Late_Indicator']].copy()
    df_assoc.dropna(inplace=True)
    transactions = df_assoc.to_numpy().tolist()
    # One-hot encoding
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions) # boolean one-hot
    df_onehot = pd.DataFrame(te_ary, columns = te.columns_)
    # run FP-Grwoth
    frequent_itemsets = fpgrowth(df_onehot, min_support = min_support, use_colnames = True)
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = min_threshold)
    
    try:
        rules.to_csv(output_path, index = False, encoding = 'utf-8-sig')
        logger.info(f"Association riles successfullt saved to CSV")
    except Exception as e:
        logger.error(f"Error saving association rules: {e}")
        raise
    return output_path