import pandas as pd

df = pd.read_csv('data/flight_data.csv')

print(df.isnull().sum())