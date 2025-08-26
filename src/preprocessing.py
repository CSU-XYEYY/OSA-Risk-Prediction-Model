import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, header=None, skiprows=1, encoding='ISO-8859-1')
    data.fillna(data.median(), inplace=True)  # Fill missing values
    object_columns = data.select_dtypes(include=['object']).columns
    encoded_data = data.copy()
    for col in object_columns:
        le = LabelEncoder()
        encoded_data[col] = le.fit_transform(encoded_data[col])
    return encoded_data