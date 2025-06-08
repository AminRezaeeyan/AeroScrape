import joblib

# Load the file
data = joblib.load('models/preprocessor.joblib')

# Now `data` holds the object that was saved
print(data)
